#include <iostream>
#include <bits/stdc++.h> 
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iomanip>
#include <string>

#include <vector>
using namespace cv;
using namespace std;


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
		cout << m << endl;
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
				cout << dSdR.size() << endl;
				double r;

				Mat BRx, Btx, BRy, Bty, jacobian_aux;
				jacobian_aux = Mat::zeros(cv::Size(3, 1), CV_64FC1);
				Btx = Mat::zeros(cv::Size(3, 1), CV_64FC1);
				Bty = Mat::zeros(cv::Size(3, 1), CV_64FC1);
				switch (dist_type)
				{
				case(0):
					cout << "case 0" << endl;
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
					cout << "case 1" << endl;
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
					cout << "r: " << r << endl;
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
					jacobian_aux.at<double>(0, 1) = (2 * f * S1 * S2 * ((k1 + 2) * k2 * pow(r, 2))) / (pow(S3, 3));
					Btx.at<double>(0, 1) = (2 * f * S1 * S2 * ((k1 + 2) * k2 * pow(r, 2))) / (pow(S3, 3));
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
					cout << "B done" << endl;

					break;
				default:
					cout << num_intrinsic << " intrinsic params are not accepted" << endl;
					return false;
					break;
				}
				//cout << B << endl;
				
			}
		}
	}
	cout << "end jacobian" << endl;
	return true;
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

	//for (int m=1;m<= num_frame)

	return true	;
}

void nextPose(int pattern_cols, int pattern_rows, float square_size, vector<Mat> rvecs, vector<Mat>tvecs,Mat camera_matrix, Mat dist_coeffs)
{
	int numberOfFrames = tvecs.size();
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
	build_Jacobian(A, B, S, corners, camera_matrix,rvecs,tvecs, dist_coeffs, pattern_cols, pattern_rows, numberOfFrames);
	A = A.t();
	B = B.t();
	Mat ACMat = Mat::eye(A.size().height, A.size().height, CV_64FC1);

	build_autocorrelation_matrix(ACMat, S, camera_matrix, rvecs, tvecs, dist_coeffs, pattern_cols, pattern_rows, numberOfFrames);

	return;
}
