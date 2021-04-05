#include <iostream>
#include <iomanip>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "camera_calibration.h"
#include "ring/image_processing.h"
#include "ring/pattern_search.h"

class CameraCalibrationRing : public CameraCalibration {
public:
	CameraCalibrationRing(VideoCapture& cap, int pattern_cols, int pattern_rows, float square_size) : CameraCalibration(cap, pattern_cols, pattern_rows, square_size) {
		load_object_points();
	}
	CameraCalibrationRing(string file_path, int pattern_cols, int pattern_rows, float square_size) : CameraCalibration(file_path, pattern_cols, pattern_rows, square_size) {
		load_object_points();
	}

	Point2f calculate_pattern_center(vector<Point2f> pattern_points) {
		int p1, p2;
		if (pattern_cols == 4 && pattern_rows == 3) {
			p1 = 5;
			p2 = 6;
		}
		else {
			p1 = 7;
			p2 = 12;
		}
		return Point2f((pattern_points[p1].x + pattern_points[p2].x) / 2.0,
			(pattern_points[p1].y + pattern_points[p2].y) / 2.0);
	}

	bool find_points_in_frame(Mat frame, vector<Point2f>& points) {
		Mat frame_gray, thresh, frame_mask, frame_original;

		int n_points = find_pattern_points(frame, w, h, points, pattern_cols, pattern_rows);
		bool result = n_points == board_points;
		frame_gray.release();
		thresh.release();
		return result;
	}


	void load_object_points() {
		Size board_size(pattern_cols, pattern_rows);
		float square_size_2 = w / pattern_cols;
		float desp_w = h * 0.2;
		float desp_h = w * 0.2;
		object_points.clear();
		object_points_image.clear();
		for (int i = 0; i < board_size.height; i++) {
			for (int j = 0; j < board_size.width; j++) {
				object_points.push_back(Point3f(float(j * square_size), float(i * square_size), 0));
				//cout<<"object: "<<float(j * square_size)<<", "<<float(i * square_size)<<endl;
				object_points_image.push_back(Point3f(float(j * square_size_2 + desp_w), float(w - (i * square_size_2 + desp_h)), 0));
				//cout<<"image: "<<float(j * square_size_2+desp_w)<<", "<<float(w- (i * square_size_2+desp_h))<<endl;
			}
		}
	}
};

bool find_points_in_frame(Mat frame, vector<Point2f>& points, int board_points, int pattern_cols, int pattern_rows) {
	Mat frame_gray, thresh, frame_mask, frame_original;
	int w = frame.rows;
	int h = frame.cols;
	int n_points = find_pattern_points(frame, w, h, points, pattern_cols, pattern_rows);
	bool result = n_points == board_points;
	frame_gray.release();
	thresh.release();
	return result;
}

Point2f calculate_pattern_center(vector<Point2f> pattern_points, int pattern_cols, int pattern_rows) {
	int p1, p2;
	if (pattern_cols == 4 && pattern_rows == 3) {
		p1 = 5;
		p2 = 6;
	}
	else {
		p1 = 7;
		p2 = 12;
	}
	return Point2f((pattern_points[p1].x + pattern_points[p2].x) / 2.0,
		(pattern_points[p1].y + pattern_points[p2].y) / 2.0);
}

bool CheckFrame(Mat frame, int** grid, int board_points, int pattern_cols, int pattern_rows, double blockSize_x, double blockSize_y, int n_rows, int n_columns, bool checkInvariants) {
	vector<Point2f> pattern_points;
	if (frame.empty()) {
		cout << "Frame vacio" << endl;
		return false;
	}

	if (!find_points_in_frame(frame.clone(), pattern_points, board_points, pattern_cols, pattern_rows)) {
		cout << "No se encontro la cantidad correcta de puntos" << endl;
		return false;
	}
	Point2f pattern_center = calculate_pattern_center(pattern_points, pattern_cols, pattern_rows);

	int x_block = floor(pattern_center.x / blockSize_x);
	int y_block = floor(pattern_center.y / blockSize_y);
	int c_x = pattern_center.x;
	int c_y = pattern_center.y;
	bool near_center = true;
	int block_radio = (blockSize_x + blockSize_y) / (2 * 3.0);
	Point2f block_center((x_block + 0.5) * blockSize_x, (y_block + 0.5) * blockSize_y);

	if (norm(Mat(Point2f(c_x, c_y)), Mat(block_center)) > block_radio) {
		near_center = false;
	}

	if (x_block == 0 || y_block == 0 || x_block == n_columns - 1 || y_block == n_rows - 1) {
		near_center = true;
	}


	if (!near_center) {
		cout << "Posicion alejada del centro" << endl;
		return false;
	}

	if (checkInvariants) {
		projective_invariants invariants(pattern_rows, pattern_cols, pattern_points);
		if (invariants.check_colinearity()) {
			if (!invariants.check_angle_cr()) {
				cout << "Correlacion cruzada de angulos sobrepasa el error" << endl;
				return false;
			}
		}
		else {
			cout << "Colinearidad sobrepasa el error" << endl;
			return false;
		}
	}

	grid[x_block][y_block]++;
	return true;
}