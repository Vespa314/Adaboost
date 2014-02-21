//注：init_sample分别是样本的不同分布
#include<iostream>
#include<math.h>
#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include"cv.h"
#include"highgui.h"
//#include"sample.h"
//#include"classifier.h"
using namespace std;
#define P_SAMPLE_NUM 100  //正样本个数
#define N_SAMPLE_NUM 300  //负样本个数
#define FEATURE_NUM 20
#define sample_num (P_SAMPLE_NUM + N_SAMPLE_NUM)
#define iter 50		//最多迭代次数
#define pi 3.14159265358979//三角函数为弧度制
#define X_MAX 300
#define Y_MAX 300

class classifier
{
public:
	double threshold;
	int feature;
	int parity;
	double error;
	int result[sample_num];
	//double alpha;
	double beta;
};

class sample
{
public:
	double x;
	double y;
	int lable;
	double eigen_value;
	double weigth;
	double result;
	int result_lebel;
};


IplImage* frame = NULL;

void init_sample(sample sample_ori[]);
void get_eigenvalue(sample sample[],int feature);
void sort_sample(sample sample_ori[],sample sample_sort[]);
void swap(sample&,sample&);
classifier weaklearner(sample sample_ori[],int feature);
classifier weaklearner2(sample sample_ori[],int feature);
int teststrong(sample sample_ori[],classifier h[],int t);
void draw(sample s[]);
void draw_distribute(classifier h[],int T,sample s[]);

int main()
{
sample sample[sample_num];
classifier h[iter];
frame = cvCreateImage(cvSize(300,300),IPL_DEPTH_8U,3);
cvNamedWindow("show",0);
cvNamedWindow("distribute",0);
init_sample(sample);
int best = 0;//记录几个弱分类器可以达到最优分类效果
int lowest_error = sample_num;
for(int i = 0;i < iter;i++)
	{
	double error = 0.5;
	classifier h_temp;
	for(int j = 0;j < FEATURE_NUM;j++)
		{
			///////////////////////////////
			h_temp = weaklearner(sample,j);
			//h_temp = weaklearner2(sample,j);
			////////////////////////////////
			if(h_temp.error < error && h_temp.error != 0)
			{
				error = h_temp.error;
				h[i] = h_temp;
				h[i].beta = error / (1 - error);
			}
		}
	double weigth_sum = 0;
	for(int k = 0;k < sample_num;k++)
	{
		if(h[i].result[k] == sample[k].lable)
			sample[k].weigth *= h[i].beta;
		weigth_sum += sample[k].weigth;
	}

	for(int k = 0;k < sample_num;k++)
	{
	sample[k].weigth /=  weigth_sum;

	}
	int boost = teststrong(sample,h,i + 1);
	cout<<i+1<<":number of sample wrong classifiered:"<<boost<<endl;
	if(!boost) 
	{
	lowest_error = 0;
	best = i+1;
	break;
	}
	if(boost < lowest_error)
		{
		 lowest_error = boost;
		 best = i+1;
		}
	//if(  boost  == 0 || i == iter)
	//	{
	//	//cout<<"final result:"<<endl;
	//	//for(int d = 0;d < sample_num;d++)
	//	//cout<<sample[d].result<<endl;
	//	 	
	//	cout<<"feacture:"<<endl;
	//	for(int k = 0;k<i;k++)
	//	cout<<h[k].feature<<endl;
	//	cout<<endl;

	//	cout<<"error:"<<endl;
	//	for(int k = 0;k<i;k++)
	//	cout<<h[k].error<<endl;
	//	cout<<endl;

	//	cout<<"thresh:"<<endl;
	//	for(int k = 0;k<i;k++)
	//	cout<<h[k].threshold<<endl;
	//	cout<<endl;

	//	cout<<"parity:"<<endl;
	//	for(int k = 0;k<i;k++)
	//	cout<<h[k].parity<<endl;
	//	cout<<endl;

	//	cout<<"beta:"<<endl;
	//	for(int k = 0;k<i;k++)
	//	cout<<h[k].beta<<endl;
	//	cout<<endl;
	//	 break;
	//	}
	cvWaitKey(50);
	}
cout<<"前"<<best<<"个分类器可以达到最好效果！！"<<endl;
cout<<"误判个数为:"<<lowest_error<<endl;
draw_distribute(h,best,sample);
cvWaitKey(0);
cvDestroyWindow("show");
return 0;
}

//void init_sample(sample sample_ori[])
//{
//for(int i = 0;i < frame->height;i++)
//	for(int j = 0;j < frame->width;j++)
//CV_IMAGE_ELEM(frame,uchar,i,3 * j) = CV_IMAGE_ELEM(frame,uchar,i,3 * j + 1) = CV_IMAGE_ELEM(frame,uchar,i,3 * j + 2)  = 255;
//double point_x[sample_num] = {};
//double point_y[sample_num] = {};
//srand(time(0));
//int counter = 0;
//int random_x,random_y;
//while(counter < P_SAMPLE_NUM)
//	{
//	random_x = rand() % 100;
//	random_y = rand() % 100;
//	point_x[counter] = random_x + 100;
//	point_y[counter] = random_y + 100;
//	cvCircle(frame,cvPoint(int(point_x[counter]),int(point_y[counter])),3, cvScalar(0,255,0), 1);
//	counter++;
//	}
//while(counter < sample_num)
//	{
//	random_x = rand() % 300;
//	random_y = rand() % 300;
//	if(!(random_x > 100 && random_x < 200 &&random_y > 100 && random_y < 200))
//	point_x[counter] = random_x;
//	point_y[counter] = random_y;
//	cvCircle(frame,cvPoint(int(point_x[counter]),int(point_y[counter])),3, cvScalar(0,0,255), 1);
//	counter++;
//	}
//
//for(int i = 0;i < sample_num;i++)
//	{
//		sample_ori[i].x = point_x[i];
//		sample_ori[i].y = point_y[i];
//		sample_ori[i].lable = i < P_SAMPLE_NUM ? 1 : 0;
//		sample_ori[i].result_lebel = 1 - sample_ori[i].lable;
//		//sample_ori[i].weigth = 1 / double(sample_num);  //正负样本权重一样
//
//		sample_ori[i].weigth = sample_ori[i].lable==1 ? 0.5 / double(P_SAMPLE_NUM) : 0.5 / double(N_SAMPLE_NUM);//正负样本权重不一样
//	}
//cvShowImage("show",frame);
//cvWaitKey(2);
//}

void init_sample(sample sample_ori[])
{
for(int i = 0;i < frame->height;i++)
	for(int j = 0;j < frame->width;j++)
CV_IMAGE_ELEM(frame,uchar,i,3 * j) = CV_IMAGE_ELEM(frame,uchar,i,3 * j + 1) = CV_IMAGE_ELEM(frame,uchar,i,3 * j + 2)  = 255;

double point_x[sample_num] = {};
double point_y[sample_num] = {};
srand(time(0));
int counter = 0;
int random_x,random_y;
while(counter < P_SAMPLE_NUM)
	{
	random_x = rand() % 100 - 50;
	random_y = rand() % 100 - 50;
	if(random_x * random_x + random_y * random_y > 2500)
		continue;
	point_x[counter] = random_x + 150;
	point_y[counter] = random_y + 150;
	cvCircle(frame,cvPoint(int(point_x[counter]),int(point_y[counter])),3, cvScalar(0,255,0), 1);
	counter++;
	}
while(counter < sample_num)
	{
	random_x = rand() % 300 - 150;
	random_y = rand() % 300 - 150;
	if(random_x * random_x + random_y * random_y < 2500)
		continue;
	point_x[counter] = random_x + 150;
	point_y[counter] = random_y + 150;
	cvCircle(frame,cvPoint(int(point_x[counter]),int(point_y[counter])),3, cvScalar(0,0,255), 1);
	counter++;
	}

for(int i = 0;i < sample_num;i++)
	{
		sample_ori[i].x = point_x[i];
		sample_ori[i].y = point_y[i];
		sample_ori[i].lable = i < P_SAMPLE_NUM ? 1 : 0;
		sample_ori[i].result_lebel = 1 - sample_ori[i].lable;
		//sample_ori[i].weigth = 1 / double(sample_num);  //正负样本权重一样

		sample_ori[i].weigth = sample_ori[i].lable==1 ? 0.5 / double(P_SAMPLE_NUM) : 0.5 / double(N_SAMPLE_NUM);//正负样本权重不一样
	}
cvShowImage("show",frame);
cvWaitKey(2);
}



void get_eigenvalue(sample sample[],int feature)
{
for(int i = 0;i < sample_num;i++)
sample[i].eigen_value = cos(pi * feature / FEATURE_NUM) * sample[i].x + sin(pi * feature / FEATURE_NUM) * sample[i].y;
}


void sort_sample(sample sample_ori[],sample sample_sort[])//升序排列
{
for(int i = 0;i < sample_num;i++)
		sample_sort[i] = sample_ori[i];

for(int pass = 1;pass < sample_num;pass++)
	{
	int work = 1;
	for(int i = 0;i < sample_num - pass;i++)
		if(sample_sort[i].eigen_value > sample_sort[i + 1].eigen_value)
		{
			swap(sample_sort[i],sample_sort[i+1]);
			work = 0;
		}
			if(work) break;
	}

}

void swap(sample &s1,sample &s2)
{
	sample s_temp;
	s_temp = s1;
	s1 = s2;
	s2 = s_temp;
}

classifier weaklearner(sample sample_ori[],int feature)
{
sample sample_sort[sample_num];
classifier h;
get_eigenvalue(sample_ori,feature);
		
double pos_weigth = 0,neg_weigth = 0;		
for(int k = 0;k < sample_num;k++)
	{
		if(sample_ori[k].lable == 1)
			pos_weigth += sample_ori[k].weigth;
		else neg_weigth += sample_ori[k].weigth;
	}		
sort_sample(sample_ori,sample_sort);

double loss_pos_weigth = 0,loss_neg_weigth = 0;
double besterror = 0.5;
int bestparity = 0;
double bestthresh = -1;

for(int k = 1;k < sample_num;k++)
	{
	if(sample_sort[k - 1].lable == 1)
		loss_pos_weigth += sample_sort[k - 1].weigth;
	else loss_neg_weigth += sample_sort[k - 1].weigth;

		if( (loss_pos_weigth + neg_weigth - loss_neg_weigth) < besterror)
			{
			besterror = loss_pos_weigth + neg_weigth - loss_neg_weigth;
			bestparity = -1;
			bestthresh = (sample_sort[k].eigen_value + sample_sort[k - 1].eigen_value) / 2;
			}
		else if(loss_neg_weigth + pos_weigth - loss_pos_weigth < besterror)
			{
			besterror = loss_neg_weigth + pos_weigth - loss_pos_weigth;
			bestparity = 1;
			bestthresh = (sample_sort[k].eigen_value + sample_sort[k - 1].eigen_value) / 2;
			}
	}
h.threshold = bestthresh;
h.error = besterror;
h.parity = bestparity;
h.feature = feature;

for(int i = 0;i < sample_num;i++)
	{
		if(h.parity * sample_ori[i].eigen_value < h.parity * h.threshold)
			h.result[i] = 1;
		else 
			h.result[i] = 0;
	}  

return h;
}



classifier weaklearner2(sample sample_ori[],int feature)
{
	classifier h;
	return h;

//
//
//
//
//PN_cum1 = ( (max(P_cum)-P_cum) + N_cum); 
//PN_cum2 = ( (max(N_cum)-N_cum) + P_cum); 
//
//[min1,thresh_ind1]= min(PN_cum1);
//[min2,thresh_ind2]= min(PN_cum2);
//
//if (min1<min2)  
//   thresh_ind=thresh_ind1;
//   PN_cum=PN_cum1;
//else
//   thresh_ind=thresh_ind2;
//   PN_cum=PN_cum2;
//end;
//
//
//
//lpn = length(PN_cum);
//thresh = ( V_sort(thresh_ind));
//%thresh = ( V_sort(thresh_ind)+V_sort(thresh_ind+1))/2;
//p = 2 *( (P_cum(thresh_ind)>N_cum(thresh_ind)) -0.5);
	
}


int teststrong(sample sample_ori[],classifier h[],int T)
{
int	error_counter = 0;
for(int i = 0;i < sample_num;i++)
	sample_ori[i].result = 0;
double egein[sample_num][FEATURE_NUM];
for(int i = 0;i < FEATURE_NUM;i++)
	{
	get_eigenvalue(sample_ori,i);
	for(int j = 0;j < sample_num;j++)
		{
			egein[j][i] = sample_ori[j].eigen_value;
		}
	}



for(int i = 0;i < sample_num;i++)
	{
		for(int j = 0;j < T;j++)
			{
			if(h[j].parity * egein[i][h[j].feature] < h[j].parity * h[j].threshold)
				sample_ori[i].result += 0.5 * log(1 / h[j].beta);
			else 
				sample_ori[i].result -= 0.5 * log(1 / h[j].beta);
			}
		if(sample_ori[i].result > 0) 
			sample_ori[i].result_lebel = 1;
		else 
			sample_ori[i].result_lebel = 0;
		if(sample_ori[i].result_lebel != sample_ori[i].lable)
			error_counter++;
	}
draw(sample_ori);
draw_distribute(h,T,sample_ori);
//cvWaitKey(500);
return error_counter;
}

void draw(sample s[])
{
for(int i = 0;i < frame->height;i++)
	for(int j = 0;j < frame->width;j++)
CV_IMAGE_ELEM(frame,uchar,i,3 * j) = CV_IMAGE_ELEM(frame,uchar,i,3 * j + 1) = CV_IMAGE_ELEM(frame,uchar,i,3 * j + 2)  = 255;


for(int i = 0;i < sample_num;i++)
	{
		if(s[i].lable == 1)
			cvCircle(frame, cvPoint(int(s[i].x),int(s[i].y)), 3, cvScalar(0,255,0), 1);
		else 
			cvCircle(frame, cvPoint(int(s[i].x),int(s[i].y)), 3, cvScalar(0,0,255), 1);
		if(s[i].lable != s[i].result_lebel)
		{
			cvLine(frame,cvPoint(int(s[i].x - 3),int(s[i].y - 3)),cvPoint(int(s[i].x + 3),int(s[i].y + 3)),cvScalar(0,0,0),1);
			cvLine(frame,cvPoint(int(s[i].x - 3),int(s[i].y + 3)),cvPoint(int(s[i].x + 3),int(s[i].y - 3)),cvScalar(0,0,0),1);
		}
	}
cvShowImage("show",frame);
cvWaitKey(2);
}


void draw_distribute(classifier h[],int T,sample s[])
{
double wrong_rate = 0;
for(int i = 0;i < frame->height;i++)
	for(int j = 0;j < frame->width;j++)
CV_IMAGE_ELEM(frame,uchar,i,3 * j) = CV_IMAGE_ELEM(frame,uchar,i,3 * j + 1) = CV_IMAGE_ELEM(frame,uchar,i,3 * j + 2) =0;

for(int i = 0;i < X_MAX;i++)
	for(int j = 0;j < Y_MAX;j++)
	{
		double result = 0;
	for(int k = 0;k < T;k++)
		{
			double egein = i * cos(h[k].feature * pi / FEATURE_NUM) + j * sin(h[k].feature * pi / FEATURE_NUM);
			if(h[k].parity * egein < h[k].parity *h[k].threshold)
				result += 0.5 * log(1 / h[k].beta);
			else
				result -= 0.5 * log(1 / h[k].beta);
		}
	if(result > 0)
		{
		CV_IMAGE_ELEM(frame,uchar,j,3 * i + 1) = 255;
		if((i - 150) * (i - 150) + (j - 150) * (j - 150) > 2500)
			wrong_rate++;
		}
	else 
		{
		CV_IMAGE_ELEM(frame,uchar,j,3 * i + 2) = 255;
		if((i - 150) * (i - 150) + (j - 150) * (j - 150) <= 2500)
			wrong_rate++;
		}
	}
	for(int i = 0;i < sample_num;i++)
		{
		if(s[i].lable == 1)
			cvCircle(frame, cvPoint(int(s[i].x),int(s[i].y)), 3, cvScalar(255,255,0), 1);
		else 
			cvCircle(frame, cvPoint(int(s[i].x),int(s[i].y)), 3, cvScalar(255,0,255), 1);
		}
	
cout<<"强分类器错误率为："<<wrong_rate / (X_MAX *Y_MAX)<<endl<<endl;
cvShowImage("distribute",frame);
cvWaitKey(2);
}



