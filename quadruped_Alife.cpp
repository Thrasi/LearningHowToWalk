//please write neural network algorithm in UpdateSensor() function and GA algorithm in Evolve() function

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ode/ode.h>
#include <drawstuff/drawstuff.h>
//#include "texturepath.h"
#include <fstream>
#include <iostream>
#include "ann.h"
#include "ga.h"
#include <typeinfo>

using namespace std;

#ifdef dDOUBLE
#define dsDrawCapsule dsDrawCapsuleD
#define dsDrawBox     dsDrawBoxD
#define dsDrawLine    dsDrawLineD
#define dsDrawCylinder    dsDrawCylinderD
#endif

#define OUTPUT_FILE "result.txt"
const int LINK_NUM=2;  // number of links
const int JT_NUM  =2;  // number of joints
const int LEG_NUM =4;  // number of legs
const int G_LENGTH = 204;	// gene length
const int POPSIZE = 6;	// population in one generation(only even number)	// 100
const int MAX_GENERATION = 4;		// 300
const double SIM_TIME = 1.0;		// 20

fstream file;

typedef struct {
  dBodyID  body;
  dGeomID  geom;
  dJointID joint;
  dReal    m,r,x,y,z; // m:weight, r:radius, x,y,z:position
} MyLink;

dWorldID      world;        // for dynamics
dSpaceID      space;        // for collision
dGeomID       ground;       // ground
dJointGroupID contactgroup; // contact group for collision
dsFunctions   fn;           // function of drawstuff

MyLink leg[LEG_NUM][LINK_NUM],torso, sensor[LEG_NUM], upset_sensor[2]; // legÅCtorso, sensors
dJointFeedback feedback[LEG_NUM], upset_feedback[2];
dJointID hoof[LEG_NUM], upset_fixed[2];

dReal  SX = 0, SY = 0, SZ = 0.40;           // initial positon of center of gravity
dReal  l1 = 0.15, l2  = 0.17;  // lenth of links
dReal  sensor_l=0.03;		//leg sensor length
dReal  sensor_r=0.03;		//leg sensor radius
dReal  upset_s_l=0.01;		//upset sensor length
dReal  upset_s_r=0.1;		//upset sensor radius 
dReal  lx = 0.45, ly= 0.25, lz = 0.10;         // body sides
dReal  r1 = 0.03, r2 = 0.03;     // leg radius
dReal  cx1 = (lx-r1)/2, cy1 = (ly+r1)/2, cz1 = l1/2;     // temporal variable

// center of joints
dReal  c_x[LEG_NUM][LINK_NUM] = {{ 0.9*cx1, 0.9*cx1},{-cx1,-cx1}, 
                                 {-cx1,-cx1},{ 0.9*cx1, 0.9*cx1}};
dReal  c_y[LEG_NUM][LINK_NUM] = {{ cy1, cy1},{ cy1, cy1},
                                 {-cy1,-cy1},{-cy1,-cy1}};
dReal  c_z[LEG_NUM][LINK_NUM] =  {{0, -l1}, {0, -l1},{0, -l1},{0, -l1}};

const double one_step = 0.015;	//time of one step  	// 0.005
const double para_K = 100000.0;	//elastic modulus
const double para_C = 1000.0;	//viscous modulus
const double M_e = 2.71828;

//node parameter in neural network
bool sensor_node[LEG_NUM][JT_NUM+2];		// 4x4
bool hidden_node[LEG_NUM][JT_NUM+2];		// 4x4
double actuator_node[LEG_NUM][JT_NUM+1];	// 4x3

//weight(initialized by a random number)
int StoH[JT_NUM+2][JT_NUM+2];	//sensor node to hidden node 		// 4x4
int HtoA[JT_NUM+2][JT_NUM+1];	//hidden node to actuator node 		// 4x3

//thresholds(initialized by a random number)
double f_sensor_low;
double a_sensor_high[JT_NUM+1];
double a_sensor_low[JT_NUM+1];

//parameter of sigmoid function(initialized by a random number)
double para_c[JT_NUM+1];

//initial position of legs(initialized by a random number)
double init_state[LEG_NUM][JT_NUM+1];

//GA parameters
double pc=0.9;	//crossover possibility
double pm=0.001;	//mutation possibility

bool **ind;		//gene information
bool **newind;	//new gene information for next generation
double result[POPSIZE];			//to record result
int pool[POPSIZE];	//mating pool(record the number of individual)

//for write logs
struct Record{
	int number;	//number of individual
	double distance;
	int StoH[JT_NUM+2][JT_NUM+2];
	int HtoA[JT_NUM+2][JT_NUM+1];
	double f_sensor_low;
	double a_sensor_high[JT_NUM+1];
	double a_sensor_low[JT_NUM+1];
	double para_c[JT_NUM+1];
	double init_state[LEG_NUM][JT_NUM+1];
};

Record record[5];	//record the best five individuals
int record_num =0;	//effective number in record

double times =0.0;

double pre_posi=0.0;	//previous position
int generation =0;	//generation number
int ind_num = 0;	//individual number
bool ifupset = false;	//is the individual upset or not



void  makeRobot(){
	dReal torso_m = 10.0;		// weight of torso
	dReal l1m = 1.0, l2m = 1.0; // weight of links
	dReal sensor_m = 0.01;	//weight of sensors

	//link position
	dReal x[LEG_NUM][LINK_NUM] = {{ 0.9*cx1, 0.9*cx1},{-cx1,-cx1},
		                            {-cx1,-cx1},{ 0.9*cx1, 0.9*cx1}};
	dReal y[LEG_NUM][LINK_NUM] = {{ cy1, cy1},{ cy1, cy1},
		                            {-cy1,-cy1},{-cy1,-cy1}};
	dReal z[LEG_NUM][LINK_NUM] = {                                  
									{-cz1,-cz1-(l1+l2)/2},{-cz1,-cz1-(l1+l2)/2},
									{-cz1,-cz1-(l1+l2)/2},{-cz1,-cz1-(l1+l2)/2}};

	dReal r[LINK_NUM]          =  { r1, r2}; // radius of links
	dReal length[LINK_NUM]     =  { l1, l2}; // length of links
	dReal weight[LINK_NUM]     =  {l1m,l2m}; // weight of links

	//axis of joint
	dReal axis_x[LEG_NUM][LINK_NUM+1] = {{ 0, 0, 0},{ 0, 0, 0},{ 0, 0, 0},{ 0, 0, 0}};
	dReal axis_y[LEG_NUM][LINK_NUM+1] = {{ 0, 1,-1},{ 0, 1,-1},{ 0, 1,-1},{ 0, 1,-1}};
	dReal axis_z[LEG_NUM][LINK_NUM+1] = {{-1, 0, 0},{-1, 0, 0},{ 1, 0, 0},{ 1, 0, 0}};

	// crate a torso
	dMass mass;
	torso.body  = dBodyCreate(world);
	dMassSetZero(&mass);
	dMassSetBoxTotal(&mass,torso_m, lx, ly, lz);
	dBodySetMass(torso.body,&mass);	
	torso.geom = dCreateBox(space,lx, ly, lz);
	dGeomSetBody(torso.geom, torso.body);
	dBodySetPosition(torso.body, SX, SY, SZ);

	// create 4 legs
	for (int i = 0; i < LEG_NUM; i++) {
		for (int j = 0; j < LINK_NUM; j++) {
			leg[i][j].body = dBodyCreate(world);
			dBodySetPosition(leg[i][j].body, SX+x[i][j], SY+y[i][j], SZ+z[i][j]);
			dMassSetZero(&mass);
			dMassSetCapsuleTotal(&mass,weight[j],3,r[j],length[j]);
			dBodySetMass(leg[i][j].body, &mass);
			leg[i][j].geom = dCreateCapsule(space,r[j],length[j]);
			dGeomSetBody(leg[i][j].geom,leg[i][j].body);
		}
	}

	// create 4 sensors on legs
	for (int i = 0; i < LEG_NUM; i++) {
		sensor[i].body = dBodyCreate(world);
		dMassSetZero(&mass);
		dMassSetCapsuleTotal(&mass,sensor_m,3,sensor_r,sensor_l);
		dBodySetMass(sensor[i].body, &mass);
		dBodySetPosition(sensor[i].body, SX+x[i][0], SY+y[i][0], SZ-l1-l2-sensor_l/2);
		sensor[i].geom = dCreateCapsule(space,sensor_r,sensor_l);
		dGeomSetBody(sensor[i].geom,sensor[i].body);
	}

	//create upset sensor(on back)
	upset_sensor[0].body = dBodyCreate(world);
	dMassSetZero(&mass);
	dMassSetCylinderTotal(&mass,sensor_m,3,upset_s_r,upset_s_l);
	dBodySetMass(upset_sensor[0].body, &mass);
	dBodySetPosition(upset_sensor[0].body, SX, SY, SZ+lz/2);
	upset_sensor[0].geom = dCreateCylinder(space,upset_s_r,upset_s_l);
	dGeomSetBody(upset_sensor[0].geom,upset_sensor[0].body);

	//create upset sensor(on belly)
	upset_sensor[1].body = dBodyCreate(world);
	dMassSetZero(&mass);
	dMassSetCylinderTotal(&mass,sensor_m,3,upset_s_r,upset_s_l);
	dBodySetMass(upset_sensor[1].body, &mass);
	dBodySetPosition(upset_sensor[1].body, SX, SY, SZ-lz/2);
	upset_sensor[1].geom = dCreateCylinder(space,upset_s_r,upset_s_l);
	dGeomSetBody(upset_sensor[1].geom,upset_sensor[1].body);

  // create links and attach legs to the torso
	//shoulder
	for (int i = 0; i < LEG_NUM; ++i) {	
		leg[i][0].joint = dJointCreateUniversal(world, 0);
		dJointAttach(leg[i][0].joint, torso.body, leg[i][0].body);

		dJointSetUniversalAnchor(leg[i][0].joint, SX+c_x[i][0], SY+c_y[i][0],SZ+c_z[i][0]);
		dJointSetUniversalAxis1(leg[i][0].joint, axis_x[i][0], axis_y[i][0], axis_z[i][0]);	//Axis1
		dJointSetUniversalAxis2(leg[i][0].joint, axis_x[i][1], axis_y[i][1], axis_z[i][1]);	//Axis2
		dJointSetUniversalParam(leg[i][0].joint, dParamLoStop1, 0);
		dJointSetUniversalParam(leg[i][0].joint, dParamHiStop1, 0.8*M_PI);
		dJointSetUniversalParam(leg[i][0].joint, dParamLoStop2,-0.5*M_PI);
		dJointSetUniversalParam(leg[i][0].joint, dParamHiStop2, 0.7*M_PI);
	}
	//knee
	for (int i = 0; i < LEG_NUM; ++i) {
		leg[i][1].joint = dJointCreateHinge(world, 0);
		dJointAttach(leg[i][1].joint, leg[i][0].body, leg[i][1].body);

		dJointSetHingeAnchor(leg[i][1].joint, SX+c_x[i][1], SY+c_y[i][1],SZ+c_z[i][1]);
		
		dJointSetHingeAxis(leg[i][1].joint, axis_x[i][2], axis_y[i][2], axis_z[i][2]);	//Axis
		dJointSetHingeParam(leg[i][1].joint, dParamLoStop, 0);
		dJointSetHingeParam(leg[i][1].joint, dParamHiStop, 0.8*M_PI);
	}
	//leg sensors
	for (int i =0; i<LEG_NUM; ++i){
		hoof[i] = dJointCreateFixed(world,0);
		dJointAttach(hoof[i], leg[i][1].body,sensor[i].body);
		dJointSetFixed(hoof[i]);

		dJointSetFeedback(hoof[i],&feedback[i]);
	}
	//upset sensor(on back)
	upset_fixed[0] = dJointCreateFixed(world,0);
	dJointAttach(upset_fixed[0], torso.body, upset_sensor[0].body);
	dJointSetFixed(upset_fixed[0]);

	dJointSetFeedback(upset_fixed[0],&upset_feedback[0]);

	//upset sensor(on belly)
	upset_fixed[1] = dJointCreateFixed(world,0);
	dJointAttach(upset_fixed[1], torso.body, upset_sensor[1].body);
	dJointSetFixed(upset_fixed[1]);

	dJointSetFeedback(upset_fixed[1],&upset_feedback[1]);

}



void drawRobot(){
	dReal r,length;
	dVector3 sides;

	// draw torso
	dsSetColor(1.3,1.3,1.3);
	dGeomBoxGetLengths(torso.geom,sides);
	dsDrawBox(dBodyGetPosition(torso.body), dBodyGetRotation(torso.body),sides);	//ï`âÊ

	// draw legs
	for (int i = 0; i < LEG_NUM; i++) {
		for (int j = 0; j < LINK_NUM; j++ ) {
			dGeomCapsuleGetParams(leg[i][j].geom, &r,&length);
			dsDrawCapsule(dBodyGetPosition(leg[i][j].body), dBodyGetRotation(leg[i][j].body),length,r);
		}
	}

	//draw leg sensors
	dsSetColor(1.,1.,1.);
	for (int i = 0; i < LEG_NUM; i++) {
		dGeomCapsuleGetParams(sensor[i].geom, &r,&length);
		dsDrawCapsule(dBodyGetPosition(sensor[i].body), dBodyGetRotation(sensor[i].body),length,r);
	}

	//draw upset sensor(on back)
//	dGeomCylinderGetParams(upset_sensor[0].geom, &r,&length);
//	dsDrawCylinder(dBodyGetPosition(upset_sensor[0].body), dBodyGetRotation(upset_sensor[0].body),length,r);

	//draw upset sensor(on belly)
//	dGeomCylinderGetParams(upset_sensor[1].geom, &r,&length);
//	dsDrawCylinder(dBodyGetPosition(upset_sensor[1].body), dBodyGetRotation(upset_sensor[1].body),length,r);

}



static void nearCallback(void *data, dGeomID o1, dGeomID o2) {
	dBodyID b1 = dGeomGetBody(o1), b2 = dGeomGetBody(o2);
	if (b1 && b2 && dAreConnectedExcluding(b1,b2,dJointTypeContact)) return;
  // if ((o1 != ground) && (o2 != ground)) return;

	static const int N = 20;
	dContact contact[N];

	int n = dCollide(o1,o2,N,&contact[0].geom,sizeof(dContact));
	if (n > 0) {
		for (int i=0; i<n; i++) {
			contact[i].surface.mode = dContactSoftERP | dContactSoftCFM;
			contact[i].surface.mu   = 100.0; 
			contact[i].surface.soft_erp = (one_step*para_K)/(one_step*para_K + para_C);		//ERP
			contact[i].surface.soft_cfm = 1.0/(one_step*para_K+para_C);		//CFM
			dJointID c = dJointCreateContact(world,contactgroup,&contact[i]);
			dJointAttach(c,b1,b2);
		}
	}
}



void PIDcontrol(double degree1, double degree2, double degree3, int legnum){
	if(legnum<0 || legnum>=LEG_NUM) return;

	static const dReal kp = 300.0 * one_step;
	static const dReal kd = 30. *one_step;
	static const dReal ki = 0.0 *one_step;
	static const dReal fMax = 4000.0 *one_step;
	static dReal tmp, diff, u, omega;

	static dReal diffsum[LEG_NUM][JT_NUM+1] = {{0.0, 0.0, 0.0},{0.0, 0.0, 0.0},{0.0, 0.0, 0.0},{0.0, 0.0, 0.0}};	//I control

	//shoulder(back and forth)
	tmp = dJointGetUniversalAngle1(leg[legnum][0].joint);
	diff = degree1 - tmp;								
	diffsum[legnum][0] += diff;		
	omega = dJointGetUniversalAngle1Rate(leg[legnum][0].joint);	
	u = kp*diff + ki*diffsum[legnum][0] - kd*omega;					
	dJointSetUniversalParam(leg[legnum][0].joint,  dParamVel1, u);	
	dJointSetUniversalParam(leg[legnum][0].joint, dParamFMax1, fMax);	

	//shoulder(up and down)
	tmp = dJointGetUniversalAngle2(leg[legnum][0].joint);	
	diff = degree2 - tmp;								
	diffsum[legnum][1] += diff;		
	omega = dJointGetUniversalAngle2Rate(leg[legnum][0].joint);	
	u = kp * diff + ki*diffsum[legnum][1] - kd * omega;								
	dJointSetUniversalParam(leg[legnum][0].joint,  dParamVel2, u);	
	dJointSetUniversalParam(leg[legnum][0].joint, dParamFMax2, fMax);	

	//knee(up and down)
	tmp = dJointGetHingeAngle(leg[legnum][1].joint);	
	diff = degree3 - tmp;						
	diffsum[legnum][2] += diff;		
	omega = dJointGetHingeAngleRate(leg[legnum][1].joint);	
	u = kp * diff + ki*diffsum[legnum][2] - kd * omega;								
	dJointSetHingeParam(leg[legnum][1].joint,  dParamVel, u);	
	dJointSetHingeParam(leg[legnum][1].joint, dParamFMax, fMax);	
}



void walk(){

	for(int i=0; i<LEG_NUM; ++i){
		if(times > 0.3 && times < 0.4){	
			dWorldSetGravity(world, 0, 0, -9.8);
			PIDcontrol(actuator_node[i][0]*0.8*M_PI, (actuator_node[i][1]*1.2-0.5)*M_PI,actuator_node[i][2]*0.8*M_PI,i);
		}else if(times >= 0.4){
			PIDcontrol(actuator_node[i][0]*0.8*M_PI, (actuator_node[i][1]*1.2-0.5)*M_PI,actuator_node[i][2]*0.8*M_PI,i);
		}else{
			PIDcontrol(init_state[i][0], init_state[i][1],init_state[i][2],i);
		}
	}

}



void UpdateSensor(){
	dJointFeedback *fb;
	dReal fx[LEG_NUM], fy[LEG_NUM], fz[LEG_NUM], force[LEG_NUM];	// 4x1
	dReal angle[LEG_NUM][JT_NUM+1];		// 4x3

	//update upset sensor
	fb = dJointGetFeedback(upset_fixed[0]);
	fz[0] = fb->f1[2];
	if(ifupset==false && fz[0]>10.0){
		ifupset=true;
	}
	fb = dJointGetFeedback(upset_fixed[1]);
	fz[0] = fb->f1[2];
	if(ifupset==false && fz[0]>10.0){
		ifupset=true;
	}


	//update angular sensor
	for(int i=0; i<LEG_NUM; ++i){
		angle[i][0] = dJointGetUniversalAngle1(leg[i][0].joint);	//shoulder(back and forward)
		angle[i][1] = dJointGetUniversalAngle2(leg[i][0].joint);	//shoulder(up and down)
		angle[i][2] = dJointGetHingeAngle(leg[i][1].joint);		//knee(up and down)
	}

	//update forse sensor
	for(int i=0; i<LEG_NUM; ++i){
		fb = dJointGetFeedback(hoof[i]);
		fx[i] = fb->f1[0];
		fy[i] = fb->f1[1];
		fz[i] = fb->f1[2];
		force[i]= pow(fx[i]*fx[i] + fy[i]*fy[i] + fz[i]*fz[i] ,0.5);
	}


	//update sensor node

	/////////////////////////
	/////////////////////////
	//you can write neural network algorithm here
	//using angle[][] array, force[] array, and threshold prameters(a_sensor_low[], a_sensor_high[], f_sensor_low)
	//to update sensor_node[][], hidden_node[][], and actuator_node[][]
	/////////////////////////
	/////////////////////////

	for (int l = 0; l < LEG_NUM; ++l) {
		updateANNLeg(angle[l], force[l], f_sensor_low, a_sensor_low, a_sensor_high, para_c,
			&StoH[0][0], &HtoA[0][0],	sensor_node[l], hidden_node[l], actuator_node[l]);
	}

}



void Evolve(){
	/////////////////////////
	/////////////////////////
	//you can write GA algorithm here
	//using result[] array and update ind[] array	POPSIZE x G_LENGTH
	/////////////////////////
	/////////////////////////
	createPopulation(result, POPSIZE, ind, G_LENGTH, 5, 0.05);
	for (int i = 0; i < POPSIZE; i++) {
		result[i] = 0;
	}
}



//create parameter from gene
void Decode(bool gene[]){	
	int num=0;
	int numi=0;
	int numj=0;
	double p;

	//weight of neural network: -3, -2, -1, 0, 0, 1, 2, 3
	for(int i=0; i<(JT_NUM+2)*(JT_NUM+2)+(JT_NUM+2)*(JT_NUM+1); ++i){
		num = 4*gene[3*i]+ 2*gene[3*i+1] + 1* gene[3*i+2];
		numi =i;
		numj =0;
		if(i<(JT_NUM+2)*(JT_NUM+2)){
			while(numi>JT_NUM+1){
				numi -= JT_NUM+2;
				numj += 1;
			}
			StoH[numj][numi]=num-4;
			if(StoH[numj][numi]<0) StoH[numj][numi]+=1;
		}else{
			numi -= (JT_NUM+2)*(JT_NUM+2);
			while(numi>JT_NUM){
				numi -= JT_NUM+1;
				numj += 1;
			}
			HtoA[numj][numi]=num-4;
			if(HtoA[numj][numi]<0) HtoA[numj][numi]+=1;
		}
	}

	//initialize thresholds
	f_sensor_low = (4*gene[84]+ 2*gene[85] + 1* gene[86]) * 10.0;
	num = 32*gene[87]+ 16*gene[88] + 8* gene[89] + 4* gene[90] + 2*gene[91] + 1*gene[92];
	a_sensor_low[0] = num *8.0 / 630.0 * M_PI;
	num = 32*gene[93]+ 16*gene[94] + 8* gene[95] + 4* gene[96] + 2*gene[97] + 1*gene[98];
	a_sensor_high[0] = num *8.0 / 630.0 * M_PI;
	if(a_sensor_low[0]>a_sensor_high[0]){
		p = a_sensor_low[0];
		a_sensor_low[0] = a_sensor_high[0];
		a_sensor_high[0] = p;
	}

	num = 32*gene[99]+ 16*gene[100] + 8* gene[101] + 4* gene[102] + 2*gene[103] + 1*gene[104];
	a_sensor_low[1] = (num *120.0 /63.0- 50.0) / 100.0 * M_PI;
	num = 32*gene[105]+ 16*gene[106] + 8* gene[107] + 4* gene[108] + 2*gene[109] + 1*gene[110];
	a_sensor_high[1] = (num *120.0 /63.0- 50.0) / 100.0 * M_PI;
	if(a_sensor_low[1]>a_sensor_high[1]){
		p = a_sensor_low[1];
		a_sensor_low[1] = a_sensor_high[1];
		a_sensor_high[1] = p;
	}

	num = 32*gene[111]+ 16*gene[112] + 8* gene[113] + 4* gene[114] + 2*gene[115] + 1*gene[116];
	a_sensor_low[2] = num *8.0 / 630.0 * M_PI;
	num = 32*gene[117]+ 16*gene[118] + 8* gene[119] + 4* gene[120] + 2*gene[121] + 1*gene[122];
	a_sensor_high[2] = num *8.0 / 630.0 * M_PI;
	if(a_sensor_low[2]>a_sensor_high[2]){
		p = a_sensor_low[2];
		a_sensor_low[2] = a_sensor_high[2];
		a_sensor_high[2] = p;
	}

	for(int i=0; i<3; ++i){
		num= 4*gene[123+3*i]+ 2*gene[124+3*i] + 1*gene[125+3*i];
		para_c[i]= num /4.0 +0.25;
	}

	for(int i=0; i<LEG_NUM; ++i){
		num = 32*gene[132+i*18]+ 16*gene[133+i*18] + 8* gene[134+i*18] + 4* gene[135+i*18] + 2*gene[136+i*18] + 1*gene[137+i*18];
		init_state[i][0]= num *8.0 / 630.0 * M_PI;
		num = 32*gene[138+i*18]+ 16*gene[139+i*18] + 8* gene[140+i*18] + 4* gene[141+i*18] + 2*gene[142+i*18] + 1*gene[143+i*18];
		init_state[i][1]= (num *120.0 /63.0- 50.0) / 100.0 * M_PI;
		num = 32*gene[144+i*18]+ 16*gene[145+i*18] + 8* gene[146+i*18] + 4* gene[147+i*18] + 2*gene[148+i*18] + 1*gene[149+i*18];
		init_state[i][2]= num *8.0 / 630.0 * M_PI;
	}
}



//initialize gene by random number
void GeneInit(bool gene[]){	
	srand((unsigned int)time(NULL));
	double x;
	for(int i =0; i<G_LENGTH; ++i){
		x = (double)rand()/(RAND_MAX);
		if(x<0.5){
			gene[i] = true;
		}else{
			gene[i] = false;
		}
	}
}




/*** Set view point and direction ***/
void start(){
	if(generation==0){
		float xyz[3] = {  1.0f,  -1.2f, 0.5f};  // View point
		float hpr[3] = {121.0f, -10.0f, 0.0f};  // View direction

#ifdef DRAWIT
			dsSetViewpoint(xyz,hpr);                // Set View point and direction
			dsSetSphereQuality(3);
			dsSetCapsuleQuality(6);
#endif

		GeneInit(ind[ind_num]);
	}

	Decode(ind[ind_num]);
	printf("\n***generation no.%d @individual no.%d ***",generation+1,ind_num+1);			
	dWorldSetGravity(world, 0, 0, -0.0);

	//initialize node of neural networks
	for(int i=0; i<LEG_NUM;++i){
		for(int j=0;j<JT_NUM+2;++j){
			sensor_node[i][j]=false;
			hidden_node[i][j]=false;
		}
		for(int j=0;j<JT_NUM+1;++j){
			actuator_node[i][j]=0.0;
		}
	}
}



static void restart(){
	times = 0.0;
	ifupset =false;

	//destroy
	for(int i =0; i<LEG_NUM; ++i){
		for(int j=0; j<LINK_NUM; ++j){
			dJointDestroy(leg[i][j].joint);
			dBodyDestroy(leg[i][j].body);
			dGeomDestroy(leg[i][j].geom);
		}
		dJointDestroy(hoof[i]);
		dBodyDestroy(sensor[i].body);
		dGeomDestroy(sensor[i].geom);
	}
	dJointDestroy(upset_fixed[0]);
	dJointDestroy(upset_fixed[1]);
	dBodyDestroy(torso.body);
	dBodyDestroy(upset_sensor[0].body);
	dBodyDestroy(upset_sensor[1].body);
	dGeomDestroy(torso.geom);
	dGeomDestroy(upset_sensor[0].geom);
	dGeomDestroy(upset_sensor[1].geom);
	dJointGroupDestroy(contactgroup);

	//reborn
	contactgroup = dJointGroupCreate(0);
	makeRobot();
	start();
}



void command(int cmd){
	switch(cmd){
	case '\033':
	case 'q':
		exit(0);
		break;
	default:
		break;
	}
}



//write logs
void WriteFile(int rank){
	file.open(OUTPUT_FILE, ios::app);
	if(! file.is_open()){
		exit(0);
	}
	
	file << "generation number:" << generation+1 << ",rank:" << rank+1
	<< ",result:" << result[record[rank].number]
	<< ",distance:" << record[rank].distance << endl;

	file << "gene information:" << endl;
	file << "StoH";
	for(int i=0; i<JT_NUM+2; ++i){
		for(int j=0; j<JT_NUM+2; ++j){
			file<< " " << record[rank].StoH[i][j];
		}
		file << endl;
	}

	file << "HtoA";
	for(int i=0; i<JT_NUM+2; ++i){
		for(int j=0; j<JT_NUM+1; ++j){
			file<< " " << record[rank].HtoA[i][j];
		}
		file << endl;
	}

	file << "f_sensor_low " << record[rank].f_sensor_low << endl;

	file << "a_sensor_low";
	for(int i=0; i<JT_NUM+1; ++i){
		file<< " " << record[rank].a_sensor_low[i];
	}
	file << endl;

	file << "a_sensor_high";
	for(int i=0; i<JT_NUM+1; ++i){
		file<< " " << record[rank].a_sensor_high[i];
	}
	file << endl;

	file << "para_c";
	for(int i=0; i<JT_NUM+1; ++i){
		file<< " " << record[rank].para_c[i];
	}
	file << endl;

	file << "init_state";
	for(int i=0; i<LEG_NUM; ++i){
		for(int j=0; j<JT_NUM+1; ++j){
			file<< " " << record[rank].init_state[i][j];
		}
		file << endl;
	}
	file << endl;

	file.close();
}



double Evaluate(double x, double y, double z){

	if(result[ind_num]<0) result[ind_num]=0;
	if(ifupset) result[ind_num]=0;
	printf("result %f\n",result[ind_num]);

	return result[ind_num];
}



void UpdateRecord(double evaluate_result, double x_distance){
	bool flag=true;
	for(int i=0; i<record_num; ++i){
		if(evaluate_result > result[record[i].number]){

			int m = record_num < 4 ? record_num : 4;
			for(m; m>i; --m){
				for(int x=0; x<JT_NUM+2; ++x){
					for(int y =0; y<JT_NUM+2; ++y){
						record[m].StoH[x][y]=record[m-1].StoH[x][y];
					}
					for(int y =0; y<JT_NUM+1; ++y){
						record[m].HtoA[x][y]=record[m-1].HtoA[x][y];
						record[m].init_state[x][y]=record[m-1].init_state[x][y];
					}
				}
				for(int x=0; x<JT_NUM+1; ++x){
					record[m].a_sensor_high[x]=record[m-1].a_sensor_high[x];
					record[m].a_sensor_low[x]=record[m-1].a_sensor_low[x];
					record[m].para_c[x]=record[m-1].para_c[x];
				}
				record[m].f_sensor_low = record[m-1].f_sensor_low;
				record[m].distance = record[m-1].distance;
				record[m].number = record[m-1].number;
			}

			for(int x=0; x<JT_NUM+2; ++x){
				for(int y =0; y<JT_NUM+2; ++y){
					record[i].StoH[x][y]=StoH[x][y];
				}
				for(int y =0; y<JT_NUM+1; ++y){
					record[i].HtoA[x][y]=HtoA[x][y];
					record[i].init_state[x][y]=init_state[x][y];
				}
			}
			for(int x=0; x<JT_NUM+1; ++x){
				record[i].a_sensor_high[x]=a_sensor_high[x];
				record[i].a_sensor_low[x]=a_sensor_low[x];
				record[i].para_c[x]=para_c[x];
			}
			record[i].f_sensor_low = f_sensor_low;
			record[i].distance = x_distance;
			record[i].number = ind_num;

			if(record_num < 5) ++record_num;
			flag = false;
			break;
		}
	}

	if(flag && record_num < 5){
		for(int x=0; x<JT_NUM+2; ++x){
			for(int y =0; y<JT_NUM+2; ++y){
				record[record_num].StoH[x][y]=StoH[x][y];
			}
			for(int y =0; y<JT_NUM+1; ++y){
				record[record_num].HtoA[x][y]=HtoA[x][y];
				record[record_num].init_state[x][y]=init_state[x][y];
			}
		}
		for(int x=0; x<JT_NUM+1; ++x){
			record[record_num].a_sensor_high[x]=a_sensor_high[x];
			record[record_num].a_sensor_low[x]=a_sensor_low[x];
			record[record_num].para_c[x]=para_c[x];
		}
		record[record_num].f_sensor_low = f_sensor_low;
		record[record_num].distance = x_distance;
		record[record_num].number = ind_num;

		++record_num;					
	}
}



void simLoop(int pause){
	static int position_count=0;

	if (!pause) {
		dSpaceCollide(space,0,&nearCallback); // collision detection
		dWorldStep(world, one_step);              // step a simulation
		times += one_step;
//		printf("Time Step : %f\n", times);
		const dReal *pos = dBodyGetPosition(torso.body);
		//printf("x=%f\n", pos[0]);
		++position_count;

		//evaluation
		if(position_count>=20){
			position_count=0;
			const dReal *posi = dBodyGetPosition(torso.body);
			if(posi[0] - pre_posi > 0.01){
				result[ind_num] += 1.0;
				pre_posi=posi[0];
			}else if(posi[0] - pre_posi < -0.01){
				result[ind_num] -= 1.0;
				pre_posi=posi[0];
			}
		}

		dJointGroupEmpty(contactgroup);       // empty jointgroup
		UpdateSensor();
		walk();                               // gait control

		//printf("%f %f\n", times, SIM_TIME);
		//generation and individual control
		if(times > SIM_TIME){	//to next individual
			const dReal *pos = dBodyGetPosition(torso.body);
			UpdateRecord(Evaluate(pos[0],pos[1],pos[2]), pos[0]);

			ind_num++;

			if(ind_num==POPSIZE){//to next generation
				for(int i=0; i<5; ++i){
					WriteFile(i);
				}

				ind_num=0;
				generation++;
				record_num = 0;
				Evolve();	//create next generation
			}

			//to next individual
			if(generation < MAX_GENERATION){
				restart();
			}else{
				exit(0);
			}

		}
	}

#ifdef DRAWIT
		drawRobot();
#endif
}



void setDrawStuff() {
	fn.version = DS_VERSION;		
	fn.start   = &start;			
	fn.step    = &simLoop;		
	fn.command = &command;			
	fn.path_to_textures = "textures";	
}



int main(int argc, char *argv[]){
	ind =new bool*[POPSIZE];
	newind =new bool*[POPSIZE];
	for(int i=0; i<POPSIZE; ++i){
		ind[i] = new bool[G_LENGTH];
		newind[i] = new bool[G_LENGTH];
	}

	dInitODE();					//initialize ODE
	setDrawStuff();
	world        = dWorldCreate();			
	space        = dHashSpaceCreate(0);		
	contactgroup = dJointGroupCreate(0);	
	ground       = dCreatePlane(space,0,0,1,0);	
	dWorldSetGravity(world, 0, 0, -0.0);	
	dWorldSetCFM(world, 1.0/(one_step*para_K+para_C)); // global CFM
	dWorldSetERP(world, (one_step*para_K)/(one_step*para_K+para_C));  // global ERP

	makeRobot();

#ifndef DRAWIT
	start();
	while (1) {
		simLoop(0);
	}
#endif
#ifdef DRAWIT
	dsSimulationLoop(argc,argv,800,480,&fn);
#endif

	dSpaceDestroy(space);
	dWorldDestroy(world);
	dCloseODE();

	for(int i=0; i<POPSIZE; ++i){
		delete[] ind[i];
		delete[] newind[i];
	}
	delete[] ind;
	delete[] newind;

	return 0;
}
