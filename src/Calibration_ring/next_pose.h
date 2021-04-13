#include <iostream>
#include <bits/stdc++.h> 
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iomanip>
#include <string>
#include <gsl/gsl_siman.h>

#include <vector>
using namespace cv;
using namespace std;

void build_Jacobian_nextpose(Mat& A_new, Mat& B_new, Mat corners, Mat camera_matrix, Mat dist_coeffs, int board_Width, int board_Height, int num_frame, Mat& x_mat) {
	int dist_type, num_intrinsic;
	double f, u, v, k1, k2;
	//Just fx (flag CALIB_FIX_ASPECT_RATIO must be set on to have fx and fy the same)
	f = camera_matrix.at<double>(0, 0);
	u = camera_matrix.at<double>(0, 2);
	v = camera_matrix.at<double>(1, 2);

	//Define the rotation matrix and translation vector of next pose

	double Rx_aux[3][3] = {
					{1,0,0},
					{0,cos(x_mat.at<double>(0,0)),-sin(x_mat.at<double>(0,0))},
					{0,sin(x_mat.at<double>(0,0)),cos(x_mat.at<double>(0,0))}
	};
	Mat Rx = Mat(3, 3, CV_64FC1, Rx_aux);
	double Ry_aux[3][3] = {
		{cos(x_mat.at<double>(0,1)),0,sin(x_mat.at<double>(0,1))},
		{0,1,0},
		{-sin(x_mat.at<double>(0,1)),0,cos(x_mat.at<double>(0,0))}
	};
	Mat Ry = Mat(3, 3, CV_64FC1, Ry_aux);
	double Rz_aux[3][3] = {
		{cos(x_mat.at<double>(0,2)),-sin(x_mat.at<double>(0,2)),0},
		{sin(x_mat.at<double>(0,2)),cos(x_mat.at<double>(0,2)),0},
		{0,0,1}
	};
	Mat Rz = Mat(3, 3, CV_64FC1, Rz_aux);

	Mat R = Rz * Ry * Rx;// Rotation matrix in the next frame
	Mat t = Mat::zeros(cv::Size(1, 3), CV_64FC1);
	t.at<double>(0, 0) = x_mat.at<double>(0, 3);
	t.at<double>(1, 0) = x_mat.at<double>(0, 4);
	t.at<double>(2, 0) = x_mat.at<double>(0, 5);

	switch (dist_coeffs.cols)
	{
	case(0):
		//no_dist = no distortion parameters
		dist_type = 0;
		num_intrinsic = 3;
		break;
	case(1):
		//radial1 = k1
		dist_type = 1;
		num_intrinsic = 4;
		k1 = dist_coeffs.at<double>(0, 0);
		break;
	case(2):
		//radial2 = k1 & k2
		dist_type = 2;
		num_intrinsic = 5;
		k1 = dist_coeffs.at<double>(0, 0);
		k2 = dist_coeffs.at<double>(0, 1);
		break;
	default:
		//no suitable
		if (dist_coeffs.cols > 2)
		{
			dist_type = 2;
			num_intrinsic = 5;
			//Discard other parameters
			k1 = dist_coeffs.at<double>(0, 0);
			k2 = dist_coeffs.at<double>(0, 1);
		}
		else {
			num_intrinsic = 0;
			dist_type = 3;
			return;
		}
		break;
	}

	Mat S_new = Mat::zeros(cv::Size(board_Width * board_Height, 3), CV_64FC1);
	//build Jacobian for next pose
	A_new = Mat::zeros(cv::Size(num_intrinsic, 2 * board_Height * board_Width), CV_64FC1);
	B_new = Mat::zeros(cv::Size(x_mat.cols, 2 * board_Height * board_Width), CV_64FC1);
	for (int i = 1; i <= board_Height; i++)
	{
		for (int j = 1; j <= board_Width; j++) {
			int pos = j + (i - 1) * board_Width;
			double S1, S2, S3, Q1, Q2, Q3, r;
			// Calculate 3D points under the camera coordinate with a new pose
			Mat aux = R * corners.col(pos - 1) + t;
			S_new.at<double>(0, pos - 1) = aux.at<double>(0, 0);
			S1 = aux.at<double>(0, 0);
			S_new.at<double>(1, pos - 1) = aux.at<double>(1, 0);
			S2 = aux.at<double>(1, 0);
			S_new.at<double>(2, pos - 1) = aux.at<double>(2, 0);
			S3 = aux.at<double>(2, 0);

			Q1 = corners.at<double>(0, pos - 1);
			Q2 = corners.at<double>(1, pos - 1);
			Q3 = corners.at<double>(2, pos - 1);
			double cross_aux[3][3] = {
					{Q1,Q3,(Q2 * -1)},
					{(Q3 * -1),Q2,Q1},
					{Q2,(Q1 * -1),Q3}
			};
			Mat cross = Mat(3, 3, CV_64FC1, cross_aux);
			Mat dSdR = R * cross;
			Mat BRx, Btx, BRy, Bty, jacobian_aux;
			jacobian_aux = Mat::zeros(cv::Size(3, 1), CV_64FC1);
			Btx = Mat::zeros(cv::Size(3, 1), CV_64FC1);
			Bty = Mat::zeros(cv::Size(3, 1), CV_64FC1);
			cout << 2 * pos - 2 << endl;
			cout << 2 * pos - 2 + 1 << endl;
			switch (num_intrinsic)
			{
			case(3):
				//no distortion parameters
				//calculate the intrinsic part in J
				A_new.at<double>(2*pos-2,0) = S1 / S3;
				A_new.at<double>(2 * pos - 2,1) = 1;
				A_new.at<double>(2 * pos - 2,2) = 0;
				A_new.at<double>(2 * pos - 2 + 1,0) = S2 / S3;
				A_new.at<double>(2 * pos - 2 + 1,1) = 1;
				A_new.at<double>(2 * pos - 2 + 1,2) = 0;

				//calculate the extrinsic part in J
				jacobian_aux.at<double>(0, 0) = f * (1 / S3);
				Btx.at<double>(0, 0) = f * (1 / S3);
				jacobian_aux.at<double>(0, 1) = 0;
				Btx.at<double>(0, 1) = 0;
				jacobian_aux.at<double>(0, 2) = (-1) * f * (S1 / pow(S3, 2));
				Btx.at<double>(0, 2) = (-1) * f * (S1 / pow(S3, 2));

				BRx = jacobian_aux * dSdR;
				//Btx = jacobian_aux;

				jacobian_aux.at<double>(0, 0) = 0;
				Bty.at<double>(0, 0) = 0;
				jacobian_aux.at<double>(0, 1) = f * (1 / S3);
				Bty.at<double>(0, 1) = f * (1 / S3);
				jacobian_aux.at<double>(0, 2) = (-1) * f * (S2 / pow(S3, 2));
				Bty.at<double>(0, 2) = (-1) * f * (S2 / pow(S3, 2));

				BRy = jacobian_aux * dSdR;
				//Bty = jacobian_aux;
				B_new.at<double>(2*pos-2,0) = BRx.at<double>(0, 0);
				B_new.at<double>(2*pos-2,1) = BRx.at<double>(0, 1);
				B_new.at<double>(2*pos-2,2) = BRx.at<double>(0, 2);
				B_new.at<double>(2*pos-2,3) = Btx.at<double>(0, 0);
				B_new.at<double>(2*pos-2,4) = Btx.at<double>(0, 1);
				B_new.at<double>(2*pos-2,5) = Btx.at<double>(0, 2);

				B_new.at<double>(2*pos-2 + 1,0) = BRy.at<double>(0, 0);
				B_new.at<double>(2*pos-2 + 1,1) = BRy.at<double>(0, 1);
				B_new.at<double>(2*pos-2 + 1,2) = BRy.at<double>(0, 2);
				B_new.at<double>(2*pos-2 + 1,3) = Bty.at<double>(0, 0);
				B_new.at<double>(2*pos-2 + 1,4) = Bty.at<double>(0, 1);
				B_new.at<double>(2*pos-2 + 1,5) = Bty.at<double>(0, 2);
				break;
			case 4:
				//K1
				r = (1 / S3) * sqrt((pow(S1, 2) + pow(S2, 2)));

				//build intrinsic parts in the Jacobian matrix
				A_new.at<double>(2 * pos - 2,0) = (1 + (k1 * pow(r, 2))) * (S1 / S3);
				A_new.at<double>(2 * pos - 2,1) = 1;
				A_new.at<double>(2 * pos - 2,2) = 0;
				A_new.at<double>(2 * pos - 2,3) = pow(r, 2) * f * (S1 / S3);
				A_new.at<double>(2 * pos - 1,0) = (1 + (k1 * pow(r, 2))) * (S1 / S3);
				A_new.at<double>(2 * pos - 1,1) = 0;
				A_new.at<double>(2 * pos - 1,2) = 1;
				A_new.at<double>(2 * pos - 1,3) = pow(r, 2) * f * (S2 / S3);

				//build extrinsic parts in the Jacobian matrix
				jacobian_aux.at<double>(0, 0) = (f / S3) + ((f * k1 * ((3 * pow(S1, 2)) + pow(S2, 2))) / pow(S3, 3));
				Btx.at<double>(0, 0) = (f / S3) + ((f * k1 * ((3 * pow(S1, 2)) + pow(S2, 2))) / pow(S3, 3));
				jacobian_aux.at<double>(0, 1) = (2 * f * k1 * S1 * S2) / (pow(S3, 3));
				Btx.at<double>(0, 1) = (2 * f * k1 * S1 * S2) / (pow(S3, 3));
				jacobian_aux.at<double>(0, 2) = ((-1) * f * (S1 / pow(S3, 2))) - (3 * f * k1 * S1 * (pow(S1, 2) + pow(S2, 2)) / (pow(S3, 4)));
				Btx.at<double>(0, 2) = ((-1) * f * (S1 / pow(S3, 2))) - (3 * f * k1 * S1 * (pow(S1, 2) + pow(S2, 2)) / (pow(S3, 4)));

				BRx = jacobian_aux * dSdR;
				//Btx = jacobian_aux;

				jacobian_aux.at<double>(0, 0) = (2 * f * k1 * S1 * S2) / (pow(S3, 3));
				Bty.at<double>(0, 0) = (2 * f * k1 * S1 * S2) / (pow(S3, 3));
				jacobian_aux.at<double>(0, 1) = (f / S3) + ((f * k1 * ((3 * pow(S2, 2)) + pow(S1, 2))) / pow(S3, 3));
				Bty.at<double>(0, 1) = (f / S3) + ((f * k1 * ((3 * pow(S2, 2)) + pow(S1, 2))) / pow(S3, 3));
				jacobian_aux.at<double>(0, 2) = ((-1) * f * (S2 / pow(S3, 2))) - (3 * f * k1 * S2 * (pow(S1, 2) + pow(S2, 2)) / (pow(S3, 4)));
				Bty.at<double>(0, 2) = ((-1) * f * (S2 / pow(S3, 2))) - (3 * f * k1 * S2 * (pow(S1, 2) + pow(S2, 2)) / (pow(S3, 4)));

				BRy = jacobian_aux * dSdR;
				//Bty = jacobian_aux;

				B_new.at<double>(2 * pos - 2,0) = BRx.at<double>(0, 0);
				B_new.at<double>(2 * pos - 2,1) = BRx.at<double>(0, 1);
				B_new.at<double>(2 * pos - 2,2) = BRx.at<double>(0, 2);
				B_new.at<double>(2 * pos - 2,3) = Btx.at<double>(0, 0);
				B_new.at<double>(2 * pos - 2,4) = Btx.at<double>(0, 1);
				B_new.at<double>(2 * pos - 2,5) = Btx.at<double>(0, 2);

				B_new.at<double>(2 * pos - 2 + 1,0) = BRy.at<double>(0, 0);
				B_new.at<double>(2 * pos - 2 + 1,1) = BRy.at<double>(0, 1);
				B_new.at<double>(2 * pos - 2 + 1,2) = BRy.at<double>(0, 2);
				B_new.at<double>(2 * pos - 2 + 1,3) = Bty.at<double>(0, 0);
				B_new.at<double>(2 * pos - 2 + 1,4) = Bty.at<double>(0, 1);
				B_new.at<double>(2 * pos - 2 + 1,5) = Bty.at<double>(0, 2);
				break;
			case 5:
				//k1 y k2
				r = (1 / S3) * sqrt((pow(S1, 2) + pow(S2, 2)));				
				//build intrinsic parts in the Jacobian matrix
				A_new.at<double>(2 * pos - 2,0) = (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) * (S1 / S3);
				A_new.at<double>(2 * pos - 2,1) = 1;
				A_new.at<double>(2 * pos - 2,2) = 0;
				A_new.at<double>(2 * pos - 2,3) = pow(r, 2) * f * (S1 / S3);
				A_new.at<double>(2 * pos - 2,4) = pow(r, 4) * f * (S1 / S3);
				A_new.at<double>(2 * pos - 2 + 1,0) = (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) * (S2 / S3);
				A_new.at<double>(2 * pos - 2 + 1,1) = 0;
				A_new.at<double>(2 * pos - 2 + 1,2) = 1;
				A_new.at<double>(2 * pos - 2 + 1,3) = pow(r, 2) * f * (S2 / S3);
				A_new.at<double>(2 * pos - 2 + 1,4) = pow(r, 4) * f * (S2 / S3);
				cout << "A done" << endl;

				//build extrinsic parts in the Jacobian matrix
				jacobian_aux.at<double>(0, 0) = (f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / S3 + (f * pow(S1, 2) * ((2 * k1) + (4 * k2 * pow(r, 2)))) / (pow(S3, 3));
				Btx.at<double>(0, 0) = (f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / S3 + (f * pow(S1, 2) * ((2 * k1) + (4 * k2 * pow(r, 2)))) / (pow(S3, 3));
				jacobian_aux.at<double>(0, 1) = (2 * f * S1 * S2 * (k1 + 2 * k2 * pow(r, 2))) / (pow(S3, 3));
				Btx.at<double>(0, 1) = (2 * f * S1 * S2 * (k1 + 2 * k2 * pow(r, 2))) / (pow(S3, 3));
				jacobian_aux.at<double>(0, 2) = (((-2) * S1 * f * pow(r, 2) * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 2))) - ((S1 * f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / (pow(S3, 2)));
				Btx.at<double>(0, 2) = (((-2) * S1 * f * pow(r, 2) * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 2))) - ((S1 * f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / (pow(S3, 2)));
				BRx = jacobian_aux * dSdR;
				//Btx = jacobian_aux;
				cout << "Bx done" << endl;

				jacobian_aux.at<double>(0, 0) = (2 * f * S1 * S2 * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 3));
				Bty.at<double>(0, 0) = (2 * f * S1 * S2 * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 3));
				jacobian_aux.at<double>(0, 1) = (f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) / S3) + (f * pow(S2, 2) * ((2 * k1) + (4 * k2 * pow(r, 2))) / pow(S3, 3));
				Bty.at<double>(0, 1) = (f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) / S3) + (f * pow(S2, 2) * ((2 * k1) + (4 * k2 * pow(r, 2))) / pow(S3, 3));
				jacobian_aux.at<double>(0, 2) = (((-2) * S2 * f * pow(r, 2) * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 2))) - ((S2 * f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / (pow(S3, 2)));
				Bty.at<double>(0, 2) = (((-2) * S2 * f * pow(r, 2) * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 2))) - ((S2 * f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / (pow(S3, 2)));

				BRy = jacobian_aux * dSdR;
				//Bty = jacobian_aux;
				cout << "By done" << endl;

				B_new.at<double>(2 * pos - 2,0) = BRx.at<double>(0, 0);
				B_new.at<double>(2 * pos - 2,1) = BRx.at<double>(0, 1);
				B_new.at<double>(2 * pos - 2,2) = BRx.at<double>(0, 2);
				B_new.at<double>(2 * pos - 2,3) = Btx.at<double>(0, 0);
				B_new.at<double>(2 * pos - 2,4) = Btx.at<double>(0, 1);
				B_new.at<double>(2 * pos - 2,5) = Btx.at<double>(0, 2);

				B_new.at<double>(2 * pos - 2 + 1,0) = BRy.at<double>(0, 0);
				B_new.at<double>(2 * pos - 2 + 1,1) = BRy.at<double>(0, 1);
				B_new.at<double>(2 * pos - 2 + 1,2) = BRy.at<double>(0, 2);
				B_new.at<double>(2 * pos - 2 + 1,3) = Bty.at<double>(0, 0);
				B_new.at<double>(2 * pos - 2 + 1,4) = Bty.at<double>(0, 1);
				B_new.at<double>(2 * pos - 2 + 1,5) = Bty.at<double>(0, 2);
				cout << "B done" << endl;
				break;
			default:
				break;
			}
		}
	}

}

bool build_Jacobian(Mat& A, Mat& B, vector<Mat> S, Mat corners, Mat camera_matrix, vector<Mat> rvecs, vector<Mat>tvecs, Mat dist_coeffs, int board_Width, int board_Height, int num_frame)
{
	int dist_type, num_intrinsic;
	double f, u, v, k1, k2;
	//Just fx (flag CALIB_FIX_ASPECT_RATIO must be set on to have fx and fy the same)
	f = camera_matrix.at<double>(0, 0);
	u = camera_matrix.at<double>(0, 2);
	v = camera_matrix.at<double>(1, 2);
	//A = (2n * m)* k
	switch (dist_coeffs.cols)
	{
	case(0):
		//no_dist = no distortion parameters
		dist_type = 0;
		num_intrinsic = 3;
		break;
	case(1):
		//radial1 = k1
		dist_type = 1;
		num_intrinsic = 4;
		k1 = dist_coeffs.at<double>(0, 0);
		break;
	case(2):
		//radial2 = k1 & k2
		dist_type = 2;
		num_intrinsic = 5;
		k1 = dist_coeffs.at<double>(0, 0);
		k2 = dist_coeffs.at<double>(0, 1);
		break;
	default:
		//no suitable
		if (dist_coeffs.cols > 2)
		{
			dist_type = 2;
			num_intrinsic = 5;
			//Discard other parameters
			k1 = dist_coeffs.at<double>(0, 0);
			k2 = dist_coeffs.at<double>(0, 1);
		}
		else {
			num_intrinsic = 0;
			dist_type = 3;
			return false;
		}
		break;
	}
	
	/*dist_type = 0;
	num_intrinsic = 3;*/
	A = Mat::zeros(cv::Size(2 * board_Width * board_Height * num_frame, num_intrinsic), CV_64FC1);
	B = Mat::zeros(cv::Size(2 * board_Width * board_Height * num_frame, 6 * num_frame), CV_64FC1);
	for (int m = 0; m < num_frame; m++) {
		//cout << m << endl;
		for (int i = 1; i <= board_Height; i++) {
			for (int j = 1; j <= board_Width; j++) {
				int pos = m * 2 * board_Width * board_Height + 2 * (j + (i - 1) * board_Width - 1);
				double S1, S2, S3, Q1, Q2, Q3;
				S1 = S[m].at<double>(0, (j + (i - 1) * board_Width) - 1);
				S2 = S[m].at<double>(1, (j + (i - 1) * board_Width) - 1);
				S3 = S[m].at<double>(2, (j + (i - 1) * board_Width) - 1);
				Q1 = corners.at<double>(0, (j + (i - 1) * board_Width) - 1);
				Q2 = corners.at<double>(1, (j + (i - 1) * board_Width) - 1);
				Q3 = corners.at<double>(2, (j + (i - 1) * board_Width) - 1);

				double cross_aux[3][3] = {
					{Q1,Q3,(Q2 * -1)},
					{(Q3 * -1),Q2,Q1},
					{Q2,(Q1 * -1),Q3}
				};
				Mat cross = Mat(3, 3, CV_64FC1, cross_aux);
				Mat dSdR = rvecs[m] * cross;
				//cout << dSdR.size() << endl;
				double r;

				Mat BRx, Btx, BRy, Bty, jacobian_aux;
				jacobian_aux = Mat::zeros(cv::Size(3, 1), CV_64FC1);
				Btx = Mat::zeros(cv::Size(3, 1), CV_64FC1);
				Bty = Mat::zeros(cv::Size(3, 1), CV_64FC1);
				switch (dist_type)
				{
				case(0):
					//cout << "case 0" << endl;
					//no distortion parameters
					//build intrinsic parts in the Jacobian matrix
					A.at<double>(0, pos) = S1/S3;
					A.at<double>(1, pos) = 1;
					A.at<double>(2, pos) = 0;
					A.at<double>(0, pos+1) = S2 / S3;
					A.at<double>(1, pos+1) = 1;
					A.at<double>(2, pos+1) = 0;

					//build extrinsic parts in the Jacobian matrix
					jacobian_aux.at<double>(0, 0) = f * (1 / S3);
					Btx.at<double>(0, 0) = f * (1 / S3);
					jacobian_aux.at<double>(0, 1) = 0;
					Btx.at<double>(0, 1) = 0;
					jacobian_aux.at<double>(0, 2) = (-1) * f * (S1 / pow(S3, 2));
					Btx.at<double>(0, 2) = (-1) * f * (S1 / pow(S3, 2));

					BRx = jacobian_aux * dSdR;
					//Btx = jacobian_aux;

					jacobian_aux.at<double>(0, 0) = 0;
					Bty.at<double>(0, 0) = 0;
					jacobian_aux.at<double>(0, 1) = f * (1 / S3);
					Bty.at<double>(0, 1) = f * (1 / S3);
					jacobian_aux.at<double>(0, 2) = (-1) * f * (S2 / pow(S3, 2));
					Bty.at<double>(0, 2) = (-1) * f * (S2 / pow(S3, 2));

					BRy = jacobian_aux * dSdR;
					//Bty = jacobian_aux;
					B.at<double>(0 + (m * 6), pos) = BRx.at<double>(0, 0);
					B.at<double>(1 + (m * 6), pos) = BRx.at<double>(0, 1);
					B.at<double>(2 + (m * 6), pos) = BRx.at<double>(0, 2);
					B.at<double>(3 + (m * 6), pos) = Btx.at<double>(0, 0);
					B.at<double>(4 + (m * 6), pos) = Btx.at<double>(0, 1);
					B.at<double>(5 + (m * 6), pos) = Btx.at<double>(0, 2);

					B.at<double>(0 + m * 6, pos + 1) = BRy.at<double>(0, 0);
					B.at<double>(1 + m * 6, pos + 1) = BRy.at<double>(0, 1);
					B.at<double>(2 + m * 6, pos + 1) = BRy.at<double>(0, 2);
					B.at<double>(3 + m * 6, pos + 1) = Bty.at<double>(0, 0);
					B.at<double>(4 + m * 6, pos + 1) = Bty.at<double>(0, 1);
					B.at<double>(5 + m * 6, pos + 1) = Bty.at<double>(0, 2);
						
					break;
				case(1):
					//K1
					//cout << "case 1" << endl;
					r = (1 / S3) * sqrt((pow(S1, 2) + pow(S2, 2)));
					
					//build intrinsic parts in the Jacobian matrix
					A.at<double>(0, pos) = (1 + (k1 * pow(r, 2))) * (S1 / S3);
					A.at<double>(1, pos) = 1;
					A.at<double>(2, pos) = 0;
					A.at<double>(3, pos) = pow(r, 2) * f * (S2 / S3);
					A.at<double>(0, pos + 1) = S2 / S3;
					A.at<double>(1, pos + 1) = 0;
					A.at<double>(2, pos + 1) = 1;
					A.at<double>(3, pos + 1) = pow(r, 2) * f * (S2 / S3);
					
					//build extrinsic parts in the Jacobian matrix
					jacobian_aux.at<double>(0, 0) = (f / S3) + ((f * k1 * ((3 * pow(S1, 2)) + pow(S2, 2))) / pow(S3, 4));
					Btx.at<double>(0, 0) = (f / S3) + ((f * k1 * ((3 * pow(S1, 2)) + pow(S2, 2))) / pow(S3, 4));
					jacobian_aux.at<double>(0, 1) = (2 * f * k1 * S1 * S2) / (pow(S3, 3));
					Btx.at<double>(0, 1) = (2 * f * k1 * S1 * S2) / (pow(S3, 3));
					jacobian_aux.at<double>(0, 2) = ((-1) * f * (S1 / pow(S3, 2))) - (3 * f * k1 * S1 * (pow(S1, 2) + pow(S2, 2)) / (pow(S3, 4)));
					Btx.at<double>(0, 2) = ((-1) * f * (S1 / pow(S3, 2))) - (3 * f * k1 * S1 * (pow(S1, 2) + pow(S2, 2)) / (pow(S3, 4)));

					BRx = jacobian_aux * dSdR;
					//Btx = jacobian_aux;

					jacobian_aux.at<double>(0, 0) = (2 * f * k1 * S1 * S2) / (pow(S3, 3));
					Bty.at<double>(0, 0) = (2 * f * k1 * S1 * S2) / (pow(S3, 3));
					jacobian_aux.at<double>(0, 1) = (f / S3) + ((f * k1 * ((3 * pow(S2, 2)) + pow(S1, 2))) / pow(S3, 3));
					Bty.at<double>(0, 1) = (f / S3) + ((f * k1 * ((3 * pow(S2, 2)) + pow(S1, 2))) / pow(S3, 3));
					jacobian_aux.at<double>(0, 2) = ((-1) * f * (S2 / pow(S3, 2))) - (3 * f * k1 * S2 * (pow(S1, 2) + pow(S2, 2)) / (pow(S3, 4)));
					Bty.at<double>(0, 2) = ((-1) * f * (S2 / pow(S3, 2))) - (3 * f * k1 * S2 * (pow(S1, 2) + pow(S2, 2)) / (pow(S3, 4)));

					BRy = jacobian_aux * dSdR;
					//Bty = jacobian_aux;

					B.at<double>(0 + (m * 6), pos) = BRx.at<double>(0, 0);
					B.at<double>(1 + (m * 6), pos) = BRx.at<double>(0, 1);
					B.at<double>(2 + (m * 6), pos) = BRx.at<double>(0, 2);
					B.at<double>(3 + (m * 6), pos) = Btx.at<double>(0, 0);
					B.at<double>(4 + (m * 6), pos) = Btx.at<double>(0, 1);
					B.at<double>(5 + (m * 6), pos) = Btx.at<double>(0, 2);

					B.at<double>(0 + m * 6, pos + 1) = BRy.at<double>(0, 0);
					B.at<double>(1 + m * 6, pos + 1) = BRy.at<double>(0, 1);
					B.at<double>(2 + m * 6, pos + 1) = BRy.at<double>(0, 2);
					B.at<double>(3 + m * 6, pos + 1) = Bty.at<double>(0, 0);
					B.at<double>(4 + m * 6, pos + 1) = Bty.at<double>(0, 1);
					B.at<double>(5 + m * 6, pos + 1) = Bty.at<double>(0, 2);

					break;
				case(2):
					//k1 y k2
					r = (1 / S3) * sqrt((pow(S1, 2) + pow(S2, 2)));
					//cout << "r: " << r << endl;
					//build intrinsic parts in the Jacobian matrix
					A.at<double>(0, pos) = (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) * (S1 / S3);
					A.at<double>(1, pos) = 1;
					A.at<double>(2, pos) = 0;
					A.at<double>(3, pos) = pow(r, 2) * f * (S1 / S3);
					A.at<double>(4, pos) = pow(r, 4) * f * (S1 / S3);
					A.at<double>(0, pos + 1) = (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) * (S2 / S3);
					A.at<double>(1, pos + 1) = 0;
					A.at<double>(2, pos + 1) = 1;
					A.at<double>(3, pos + 1) = pow(r, 2) * f * (S2 / S3);
					A.at<double>(4, pos + 1) = pow(r, 4) * f * (S2 / S3);

					//build extrinsic parts in the Jacobian matrix
					jacobian_aux.at<double>(0, 0) = (f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / S3 + (f * pow(S1, 2) * ((2 * k1) + (4 * k2 * pow(r, 2)))) / (pow(S3, 3));
					Btx.at<double>(0, 0) = (f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / S3 + (f * pow(S1, 2) * ((2 * k1) + (4 * k2 * pow(r, 2)))) / (pow(S3, 3));
					jacobian_aux.at<double>(0, 1) = (2 * f * S1 * S2 * (k1 + 2 * k2 * pow(r, 2))) / (pow(S3, 3));
					Btx.at<double>(0, 1) = (2 * f * S1 * S2 * (k1 + 2 * k2 * pow(r, 2))) / (pow(S3, 3));
					jacobian_aux.at<double>(0, 2) = (((-2) * S1 * f * pow(r, 2) * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 2))) - ((S1 * f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / (pow(S3, 2)));
					Btx.at<double>(0, 2) = (((-2) * S1 * f * pow(r, 2) * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 2))) - ((S1 * f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / (pow(S3, 2)));
					BRx = jacobian_aux * dSdR;
					//Btx = jacobian_aux;
					//cout << "Bx done" << endl;

					jacobian_aux.at<double>(0, 0) = (2 * f * S1 * S2 * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 3));
					Bty.at<double>(0, 0) = (2 * f * S1 * S2 * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 3));
					jacobian_aux.at<double>(0, 1) = (f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) / S3) + (f * pow(S2, 2) * ((2 * k1) + (4 * k2 * pow(r, 2))) / pow(S3, 3));
					Bty.at<double>(0, 1) = (f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) / S3) + (f * pow(S2, 2) * ((2 * k1) + (4 * k2 * pow(r, 2))) / pow(S3, 3));
					jacobian_aux.at<double>(0, 2) = (((-2) * S2 * f * pow(r, 2) * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 2))) - ((S2 * f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / (pow(S3, 2)));
					Bty.at<double>(0, 2) = (((-2) * S2 * f * pow(r, 2) * (k1 + (2 * k2 * pow(r, 2)))) / (pow(S3, 2))) - ((S2 * f * (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4)))) / (pow(S3, 2)));

					BRy = jacobian_aux * dSdR;
					//Bty = jacobian_aux;
					//cout << "By done" << endl;

					B.at<double>(0 + (m * 6), pos) = BRx.at<double>(0, 0);
					B.at<double>(1 + (m * 6), pos) = BRx.at<double>(0, 1);
					B.at<double>(2 + (m * 6), pos) = BRx.at<double>(0, 2);
					B.at<double>(3 + (m * 6), pos) = Btx.at<double>(0, 0);
					B.at<double>(4 + (m * 6), pos) = Btx.at<double>(0, 1);
					B.at<double>(5 + (m * 6), pos) = Btx.at<double>(0, 2);

					B.at<double>(0 + m * 6, pos + 1) = BRy.at<double>(0, 0);
					B.at<double>(1 + m * 6, pos + 1) = BRy.at<double>(0, 1);
					B.at<double>(2 + m * 6, pos + 1) = BRy.at<double>(0, 2);
					B.at<double>(3 + m * 6, pos + 1) = Bty.at<double>(0, 0);
					B.at<double>(4 + m * 6, pos + 1) = Bty.at<double>(0, 1);
					B.at<double>(5 + m * 6, pos + 1) = Bty.at<double>(0, 2);
					//cout << "B done" << endl;

					break;
				default:
					//cout << num_intrinsic << " intrinsic params are not accepted" << endl;
					return false;
					break;
				}
				//cout << B << endl;
				
			}
		}
	}
	//cout << "end jacobian" << endl;
	return true;
}
double ang2cornerness(double x)
{
	double res;
	res = 1404.50760 * pow(x, 2) - 49.2631560 * pow(x, 3) + 0.94482 * pow(x, 4) - 0.0093798 * pow(x, 5) + 0.0000455668 * pow(x, 6) - 8.6160 * 1e-8 * pow(x, 7);
	return res;
}
bool build_autocorrelation_matrix(Mat& ACMat, vector<Mat> S, Mat camera_matrix, vector<Mat> rvecs, vector<Mat>tvecs, Mat dist_coeffs, int board_Width, int board_Height, int num_frame)
{
	int dist_type, num_intrinsic;
	double f, u, v, k1, k2, k3, k4;
	//Just fx (flag CALIB_FIX_ASPECT_RATIO must be set on to have fx and fy the same)
	f = camera_matrix.at<double>(0, 0);
	u = camera_matrix.at<double>(0, 2);
	v = camera_matrix.at<double>(1, 2);
	//A = (2n * m)* k
	switch (dist_coeffs.cols)
	{
	case(0):
		//no_dist = no distortion parameters
		dist_type = 0;
		num_intrinsic = 3;
		break;
	case(1):
		//radial1 = k1
		dist_type = 1;
		num_intrinsic = 4;
		k1 = dist_coeffs.at<double>(0, 0);
		break;
	case(2):
		//radial2 = k1 & k2
		dist_type = 2;
		num_intrinsic = 5;
		k1 = dist_coeffs.at<double>(0, 0);
		k2 = dist_coeffs.at<double>(0, 1);
		break;
	default:
		//no suitable
		if (dist_coeffs.cols > 2)
		{
			dist_type = 2;
			num_intrinsic = 5;
			//Discard other parameters
			k1 = dist_coeffs.at<double>(0, 0);
			k2 = dist_coeffs.at<double>(0, 1);
		}
		else {
			num_intrinsic = 0;
			dist_type = 3;
			return false;
		}
		break;
	}

	vector<Mat> P;

	for (int m = 0; m < num_frame; m++) {
		Mat aux = Mat::zeros(cv::Size(board_Height * board_Width, 2), CV_64FC1);
		P.push_back(aux);
		for (int i = 1; i <= board_Height; i++) {
			for (int j = 1; j <= board_Width; j++) {
				int pos = -1 + j + (i - 1) * board_Width;
				double S1, S2, S3;
				S1 = S[m].at<double>(0, pos);
				S2 = S[m].at<double>(1, pos);
				S3 = S[m].at<double>(2, pos);

				double x_, y_, r, theta, theta_d;

				switch (num_intrinsic)
				{
				case 3:
					//no distortion
					x_ = S1 / S3;
					y_ = S2 / S3;
					P[m].at<double>(0, pos) = f * x_ + u;
					P[m].at<double>(1, pos) = f * y_ + v;
					break;
				case 4:
					//k1
					r = (1 / S3) * sqrt(pow(S1, 2) + pow(S2, 2));
					x_ = S1 / S3;
					y_ = S2 / S3;
					P[m].at<double>(0, pos) = (1 + (k1 * pow(r, 2))) * f * x_ + u;
					P[m].at<double>(1, pos) = (1 + (k1 * pow(r, 2))) * f * y_ + v;
					break;
				case 5:
					//k2
					r = (1 / S3) * sqrt(pow(S1, 2) + pow(S2, 2));
					x_ = S1 / S3;
					y_ = S2 / S3;
					P[m].at<double>(0, pos) = (1 + (k1 * pow(r, 2) + (k2 * pow(r, 4)))) * f * x_ + u;
					P[m].at<double>(1, pos) = (1 + (k1 * pow(r, 2) + (k2 * pow(r, 4)))) * f * y_ + v;
					break;
				case 7:
					//fisheye
					r = (1 / S3) * sqrt(pow(S1, 2) + pow(S2, 2));
					theta = atan(r);
					theta_d = theta + (k1 * pow(theta, 3)) + (k2 * pow(theta, 5)) + (k3 * pow(theta, 7)) + (k4 * pow(theta, 9));
					P[m].at<double>(0, pos) = u + theta_d * f * S1 / (r * S3);
					P[m].at<double>(1, pos) = v + theta_d * f * S2 / (r * S3);
					break;
				default:
					break;
				}
			}
		}
	}

	ACMat = Mat::zeros(cv::Size(2 * board_Height * board_Width * num_frame, 2 * board_Height * board_Width * num_frame), CV_64FC1);

	for (int m = 1; m <= num_frame; m++) {
		int initial_pose, final_pose, act_frame;
		act_frame = m - 1;
		initial_pose = (m - 1) * 2 * board_Height * board_Width;
		final_pose = (m * 2 * board_Height * board_Width) - 1;
		Mat heightVec, widthVec;
		for (int i = 1; i <= board_Height; i++) {
			for (int j = 1; j <= board_Width; j++) {
				int pos = (j + (i - 1) * board_Width) - 1;
				if (j != board_Width)
				{
					heightVec = P[act_frame].col(pos + 1) - P[act_frame].col(pos);
				}
				else {
					heightVec = P[act_frame].col(pos) - P[act_frame].col(pos - 1);
				}
				if (i != board_Height) {
					widthVec = P[act_frame].col(pos + board_Width) - P[act_frame].col(pos);
				}
				else {
					widthVec = P[act_frame].col(pos) - P[act_frame].col(pos - board_Width);
				}
				double heightNormal, widthNormal, dot, ang, s, AC_cur_max, AC_cur_min;
				heightNormal = norm(heightVec);
				widthNormal = norm(widthVec);
				Mat heightRes, widthRes, v, R_cur, AC_cur, aux_mat;
				heightRes = heightVec / heightNormal;
				widthRes = widthVec / widthNormal;
				R_cur = Mat::zeros(cv::Size(2, 2), CV_64FC1);
				AC_cur = Mat::zeros(cv::Size(2, 2), CV_64FC1);
				dot = heightRes.dot(widthRes);
				ang = acos(dot) * 180 / CV_PI;
				if (ang > 90) {
					v = heightRes + widthRes;
				}
				else {
					v = heightRes - widthRes;
				}
				v = v / norm(v);
				R_cur.at<double>(0, 0) = v.at<double>(0, 0);
				R_cur.at<double>(0, 1) = v.at<double>(1, 0) * -1;
				R_cur.at<double>(1, 0) = v.at<double>(1, 0);
				R_cur.at<double>(1, 1) = v.at<double>(0, 0);
				s = 5e+5;
				AC_cur_max = max(ang2cornerness(180 - ang), ang2cornerness(ang));
				AC_cur_min = min(ang2cornerness(180 - ang), ang2cornerness(ang));
				AC_cur.at<double>(0, 0) = AC_cur_max / s;
				AC_cur.at<double>(0, 1) = 0;
				AC_cur.at<double>(1, 0) = 0;
				AC_cur.at<double>(1, 1) = AC_cur_min / s;
				aux_mat = R_cur * AC_cur * (R_cur.t());
				pos = initial_pose + (pos * 2);
				ACMat.at<double>(pos, pos) = aux_mat.at<double>(0, 0);
				ACMat.at<double>(pos, pos + 1) = aux_mat.at<double>(0, 1);
				ACMat.at<double>(pos + 1, pos) = aux_mat.at<double>(1, 0);
				ACMat.at<double>(pos + 1, pos + 1) = aux_mat.at<double>(1, 1);
			}
		}

	}


	return true	;
}

/* set up parameters for this simulated annealing run */

#define N_TRIES 200             /* how many points do we try before stepping */
#define ITERS_FIXED_T 2000      /* how many iterations for each T? */
#define STEP_SIZE 1.0           /* max step size in random walk */
#define K 1.0                   /* Boltzmann constant */
#define T_INITIAL 5000.0        /* initial temperature */
#define MU_T 1.002              /* damping factor for temperature */
#define T_MIN 5.0e-1

gsl_siman_params_t params = { N_TRIES, ITERS_FIXED_T, STEP_SIZE,
							 K, T_INITIAL, MU_T, T_MIN };

double E1(void* xp)
{
	// x[0] = x_init
	// x[1] = A
	// x[2] = B
	// x[3] = corners
	// x[4] = dist_coeffs
	// x[5] = basic_info (cols,rows,#frames)
	// x[6] = ACMat
	// x[7] = camera_matrix
	vector<Mat>* x = ((vector<Mat>*)xp);
	//double y = x[0];
	Mat x_mat = x->at(0);
	Mat A = x->at(1);
	Mat B = x->at(2);
	Mat corners = x->at(3);
	Mat dist_coeffs = x->at(4);
	Mat basic_info = x->at(5);
	Mat ACMat = x->at(6);
	Mat camera_matrix = x->at(7);

	int board_Width, board_Height, numberOfFrames;
	board_Width = basic_info.at<double>(0, 0);
	board_Height = basic_info.at<double>(0, 1);
	numberOfFrames = basic_info.at<double>(0, 2);
	
	double Rx_aux[3][3] = {
					{1,0,0},
					{0,cos(x_mat.at<double>(0,0)),-sin(x_mat.at<double>(0,0))},
					{0,sin(x_mat.at<double>(0,0)),cos(x_mat.at<double>(0,0))}
	};
	Mat Rx = Mat(3, 3, CV_64FC1, Rx_aux);
	double Ry_aux[3][3] = {
		{cos(x_mat.at<double>(0,1)),0,sin(x_mat.at<double>(0,1))},
		{0,1,0},
		{-sin(x_mat.at<double>(0,1)),0,cos(x_mat.at<double>(0,0))}
	};
	Mat Ry = Mat(3, 3, CV_64FC1, Ry_aux);
	double Rz_aux[3][3] = {
		{cos(x_mat.at<double>(0,2)),-sin(x_mat.at<double>(0,2)),0},
		{sin(x_mat.at<double>(0,2)),cos(x_mat.at<double>(0,2)),0},
		{0,0,1}
	};
	Mat Rz = Mat(3, 3, CV_64FC1, Rz_aux);

	Mat R = Rz * Ry * Rx; //Rotation matrix in the next frame
	Mat t = Mat::zeros(cv::Size(1, 3), CV_64FC1);
	t.at<double>(0, 0) = x_mat.at<double>(0,3);
	t.at<double>(1, 0) = x_mat.at<double>(0,4);
	t.at<double>(2, 0) = x_mat.at<double>(0,5);

	Mat P = Mat::zeros(cv::Size(board_Height * board_Width, 2), CV_64FC1);
	Mat S_new = Mat::zeros(cv::Size(board_Height * board_Width, 3), CV_64FC1);

	int dist_type, num_intrinsic, dist_border;
	dist_border = 30;
	double f, u, v, k1, k2, k3, k4, S1, S2, S3, x_, y_, r;
	bool OUT_OF_RANGE = false;
	//Just fx (flag CALIB_FIX_ASPECT_RATIO must be set on to have fx and fy the same)
	f = camera_matrix.at<double>(0, 0);
	u = camera_matrix.at<double>(0, 2);
	v = camera_matrix.at<double>(1, 2);
	for (int i = 1; i <= board_Height; i++) {
		for (int j = 1; j <= board_Width; j++) {
			int pos = j + (i - 1) * board_Width;
			//calculate 3D points under the camera coordinate with a new pose
			Mat aux = R * corners.col(pos - 1) + t;
			S_new.at<double>(0, pos-1) = aux.at<double>(0, 0);
			S1 = aux.at<double>(0, 0);
			S_new.at<double>(1, pos-1) = aux.at<double>(1, 0);
			S2 = aux.at<double>(1, 0);
			S_new.at<double>(2, pos-1) = aux.at<double>(2, 0);
			S3 = aux.at<double>(2, 0);
			switch (dist_coeffs.cols)
			{
			case(0):
				//no_dist = no distortion parameters
				num_intrinsic = 3;
				x_ = S1 / S3;
				y_ = S2 / S3;
				P.at<double>(0, pos - 1) = f * x_ + u;
				P.at<double>(1, pos - 1) = f * y_ + v;
				break;
			case(1):
				//radial1 = k1
				num_intrinsic = 4;
				k1 = dist_coeffs.at<double>(0, 0);
				r = (1 / S3) * sqrt(pow(S1, 2) + pow(S2, 2));
				x_ = S1 / S3;
				y_ = S2 / S3;
				P.at<double>(0, pos - 1) = (1 + (k1 * pow(r, 2))) * f * x_ + u;
				P.at<double>(1, pos - 1) = (1 + (k1 * pow(r, 2))) * f * y_ + v;
				break;
			case(2):
				//radial2 = k1 & k2
				num_intrinsic = 5;
				k1 = dist_coeffs.at<double>(0, 0);
				k2 = dist_coeffs.at<double>(0, 1);
				r = (1 / S3) * sqrt(pow(S1, 2) + pow(S2, 2));
				x_ = S1 / S3;
				y_ = S2 / S3;
				P.at<double>(0, pos - 1) = (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) * f * x_ + u;
				P.at<double>(1, pos - 1) = (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) * f * y_ + v;
				break;
			default:
				//no suitable
				if (dist_coeffs.cols > 2)
				{
					num_intrinsic = 5;
					//Discard other parameters
					k1 = dist_coeffs.at<double>(0, 0);
					k2 = dist_coeffs.at<double>(0, 1);
					r = (1 / S3) * sqrt(pow(S1, 2) + pow(S2, 2));
					x_ = S1 / S3;
					y_ = S2 / S3;
					P.at<double>(0, pos - 1) = (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) * f * x_ + u;
					P.at<double>(1, pos - 1) = (1 + (k1 * pow(r, 2)) + (k2 * pow(r, 4))) * f * y_ + v;
				}
				else {
					num_intrinsic = 0;
					return false;
				}
				break;
			}
			if (P.at<double>(0, pos - 1) < dist_border || P.at<double>(0, pos - 1) > 640 - dist_border) {
				OUT_OF_RANGE = true;
				break;
			} 
			if (P.at<double>(1, pos - 1) < dist_border || P.at<double>(1, pos - 1) > 480 - dist_border) {
				OUT_OF_RANGE = true;
				break;
			}
		}		
		if (OUT_OF_RANGE) {
			return DBL_MAX;
			break;
		}
	}
	Mat A_new, B_new;
	build_Jacobian_nextpose(A_new, B_new, corners, camera_matrix, dist_coeffs, board_Width, board_Height, numberOfFrames, x_mat);
	cout << "asd" << endl;
	//return exp(-pow((y - 1.0), 2.0)) * sin(8 * y);
 	return 1;
}

double M1(void* xp, void* yp)
{
	cout << "M1" << endl;
	double *x = ((double*)xp);
	double *y = ((double*)yp);

	return fabs(x[0] - y[0]);
}

void S1(const gsl_rng* r, void* xp, double step_size)
{
	cout << "S1" << endl;
	double *old_x = ((double*)xp);
	/*double *new_x;
	new_x = old_x;*/
	double u = gsl_rng_uniform(r);
	old_x[0] = u * 2 * step_size - step_size + old_x[0];

	//memcpy(xp, &new_x, sizeof(new_x));
}

void P1(void* xp)
{
	cout << "P1" << endl;
	unsigned int i;
	double* matrix = (double*)xp;
	printf("  [");
	for (i = 0; i < 6; ++i) {
		printf(" %12g ", matrix[i]);
	}
	printf("]  ");
}

void nextPose(int pattern_cols, int pattern_rows, float square_size, vector<Mat> rvecs, vector<Mat>tvecs,Mat camera_matrix, Mat dist_coeffs)
{
	int numberOfFrames = tvecs.size();
	double tranlation_bound;
	tranlation_bound = 1000;
	Mat corners = Mat::zeros(cv::Size(pattern_cols * pattern_rows, 3), CV_64FC1);
	vector<Mat> S;
	//Build 3D control points in real world coordinates
	for (int i = 1; i <= pattern_cols; i++)
	{
		for (int j = 1; j <= pattern_rows; j++)
		{
			int first_coord = -1 + j + ((i - 1) * pattern_rows);
			corners.at<double>(0, first_coord) = (j - 1) * square_size;
			corners.at<double>(1, first_coord) = (i - 1) * square_size;
		}
	}
	//corners = corners.t();
	//Translate real world points into camera points
	//S = R * Q + t
	for (int m = 0; m < numberOfFrames; m++) {
		Mat aux= Mat::zeros(cv::Size(pattern_cols * pattern_rows, 3), CV_64FC1);
		S.push_back(aux);
		Mat outr;
		Rodrigues(rvecs[m].t(), outr);
		rvecs[m] = outr;
		for (int i = 1; i <= pattern_rows; i++) {
			for (int j = 1; j <= pattern_cols; j++) {
				int location = j - 1 + ((i - 1) * pattern_cols);
				Mat actual_point;		
				actual_point = (outr * corners.col(location)) + tvecs[m];
				S[m].at<double>(0, location) = actual_point.at<double>(0, 0);
				S[m].at<double>(1, location) = actual_point.at<double>(1, 0);
				S[m].at<double>(2, location) = actual_point.at<double>(2, 0);
			}
		}
	}
	Mat A,B;
	cout << "Build Jacobian" << endl;
	build_Jacobian(A, B, S, corners, camera_matrix,rvecs,tvecs, dist_coeffs, pattern_cols, pattern_rows, numberOfFrames);
	A = A.t();
	B = B.t();
	Mat ACMat = Mat::eye(A.size().height, A.size().height, CV_64FC1);
	vector<Mat> x_init;
	cout << "autocorrelation" << endl;
	build_autocorrelation_matrix(ACMat, S, camera_matrix, rvecs, tvecs, dist_coeffs, pattern_rows, pattern_cols, numberOfFrames);
	double mean_t1, mean_t2, mean_t3;
	mean_t1 = 0;
	mean_t2 = 0;
	mean_t3 = 0;
	for (int m = 0; m < numberOfFrames; m++)
	{
		mean_t1 += tvecs[m].at<double>(0, 0);
		mean_t2 += tvecs[m].at<double>(1, 0);
		mean_t3 += tvecs[m].at<double>(2, 0);
	}
	mean_t1 /= numberOfFrames;
	mean_t2 /= numberOfFrames;
	mean_t3 /= numberOfFrames;
	double x_aux[6] = { 0,0,0,mean_t1,mean_t2,mean_t3 };
	Mat x = Mat(1, 6, CV_64FC1, x_aux);
	double lb_aux[6] = { 0,0,0,(-1 * tranlation_bound),(-1 * tranlation_bound),0 };
	Mat lb = Mat(1, 6, CV_64FC1, lb_aux);
	double ub_aux[6] = { CV_2PI,CV_PI,CV_2PI,tranlation_bound,tranlation_bound,mean_t3 };
	Mat ub = Mat(1, 6, CV_64FC1, ub_aux);
	double basic_info_aux[3] = { pattern_cols,pattern_rows,numberOfFrames };
	Mat basic_info = Mat(1, 3, CV_64FC1, basic_info_aux);
	//cost_function(x,A,B, corners, intrinsicPara, basicInfo, ACMat_extend)
	x_init.push_back(x);
	x_init.push_back(A);
	x_init.push_back(B);
	x_init.push_back(corners);
	x_init.push_back(dist_coeffs);
	x_init.push_back(basic_info);
	x_init.push_back(ACMat);
	x_init.push_back(camera_matrix);
	cout << "optimizador" << endl;
	const gsl_rng_type* T;
	gsl_rng* r;

	//double x_initial = 15.5;
	double x_initial[6] = { 0,0,0,15.8760,0.3467,71.5890 };
	//Mat x_initial = Mat(1, 6, CV_64FC1, x_aux);
	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);

	gsl_siman_solve(r, &x_init, E1, S1, M1, P1,
		NULL, NULL, NULL,
		sizeof(x_init), params);

	gsl_rng_free(r);

	return;
}
