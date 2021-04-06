#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <direct.h>
#include "camera_calibration_ring.h"
#include <fstream>

#include "next_pose.h"

#define INTRO_KEY 13
#define ESC_KEY 27
#define K_KEY 107
#define C_KEY 99

using namespace std;
using namespace cv;
vector<string> videos = {
    "hp.mkv",
    "genius.mkv",
    "s3eyecam_5_4.avi",
    "cerca.mkv",
    "medio.mkv",
    "lejos.mkv",
    "logitech.mkv"
};
string actualVid;
/**
 * @brief Initialize windows names, sizes and positions.
 */
void window_setup() {
    int window_w = 180 * 1.5;
    int window_h = 120 * 1.5;
    int second_screen_offset = 0;//1360;
    string window_name;
    window_name = "CentersDistribution";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, 0 + second_screen_offset, 0);

    window_name = "CalibrationFrames";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w + second_screen_offset, 0);

    window_name = "Undistort";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 2 + second_screen_offset, 0);

    window_name = "FrontoParallel";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, 0 + second_screen_offset, window_h + 60);

    window_name = "Reproject";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w + second_screen_offset, window_h + 60);

    window_name = "Distort";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 2 + second_screen_offset, window_h + 60);

    window_name = "Threshold";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 3 + second_screen_offset, 0);

    window_name = "Line Representation";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 3 + second_screen_offset, window_h + 60);

    window_name = "Ellipses";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 4 + second_screen_offset, 0);

    window_name = "Contours";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 4 + second_screen_offset, window_h + 60);

    window_name = "Colinearity";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 4 + second_screen_offset, window_h*2 + 60);

    window_name = "Angles Cross Relation";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 3 + second_screen_offset, window_h*2 + 60);
}

int main( int argc, char** argv ) {
	string path;
	#ifdef _WIN32
		char* buffer;

		// Get the current working directory:
		if ((buffer = _getcwd(NULL, 0)) == NULL)
			perror("_getcwd error");
		else
		{
			path = string(buffer);
			replace(path.begin(), path.end(), '/','\\');
			free(buffer);
		}
	#else	
	#endif
    /*window_setup();

    int pattern_rows = 4;
    int pattern_cols = 5;
    float square_size = 44.3;

    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/hp.mkv");
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/genius.mkv");
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/s3eyecam_5_4.avi");
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/cerca.mkv");
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/medio.mkv");
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/lejos.mkv");
    
    //for(int i = 0; i < videos.size(); i++)
    int i = 0;
    {
        actualVid = videos[i];
        
        //string vidFile = "/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/"+videos[i];
        //string vidFile = "/home/manuel/Documents/Projects/OpenCV/Rodrigo_Tesis/seminario-tesis/data/"+videos[i];
		string vidFile = "C:/Users/u1/Documents/tesis/ring_calibration_videos/" + videos[i];
        


        VideoCapture cap(vidFile);
        if ( !cap.isOpened() ) {
            cout << "Cannot open the video file. \n";
            return -1;
        }
        
        
        //int pattern_rows = 3;
        //int pattern_cols = 4;
        //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/lifecam_4_3out.mp4");
        //float square_size = 55;
        //if ( !cap.isOpened() ) {
            //cout << "Cannot open the video file. \n";
            //return -1;
        //}
        

        int n_frames  = 80;
        int grid_cols = 8;
        int grid_rows = 8;

		//Instanciacion de la clase de calibraci�n
        //CameraCalibrationRing camera_calibration(cap, pattern_cols, pattern_rows, square_size);
		CameraCalibrationRing camera_calibration("C:/Users/u1/Documents/tesis/bin/Release/frames.txt", pattern_cols, pattern_rows, square_size);
		
        // read image pathts from frame_list.txt
        //CameraCalibrationRing camera_calibration("ring_calibration_frames_full/frame_list.txt", pattern_cols, pattern_rows, square_size);
        camera_calibration.calibrate_camera_iterative(10, n_frames, grid_rows, grid_cols,actualVid);

    }
        waitKey(0);*/
	
	//Mantendremos la grilla aqui

	int n_frames = 80;
	int n_columns = 4;
	int n_rows = 4;
	int pattern_rows = 4;
	int pattern_cols = 5;
	//float square_size = 44.3;
	float square_size = 5;
	int** quadBins = new int* [n_rows];
	for (int y_block = 0; y_block < n_rows; ++y_block) {
		quadBins[y_block] = new int[n_columns];
		for (int x_block = 0; x_block < n_columns; ++x_block) {
			quadBins[y_block][x_block] = 0;
		}
	}
	
	VideoCapture cap;
	int apiID = cv::CAP_ANY;      
	int deviceID = 1;      
	int frame_cont = 0;
	cap.open(deviceID, apiID);
	Mat frame;
	double height = cap.get(CAP_PROP_FRAME_HEIGHT);
	double width = cap.get(CAP_PROP_FRAME_WIDTH);	
	double blockSize_y = height / n_rows;
	double blockSize_x = width / n_columns;
	double block_radio = (blockSize_x + blockSize_y) / (2 * 3.0);
	bool checkInvariants = false;
	
	std::ofstream images;
	images.open("images.txt");
	images.flush();
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}
	for (;;)
	{
		cap.read(frame);
		Mat showFrame = frame.clone();
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		for (int i = 0; i < n_columns; i++) {
			for (int j = 0; j < n_rows; j++) {
				Scalar GridColor = Scalar(0, 0, 255);
				int lineWidth = 1;
				if (quadBins[i][j] != 0) {
					GridColor = Scalar(0, 255, 0);
					lineWidth = 2;
				}
				rectangle(showFrame, Point(blockSize_x * i, blockSize_y * j), Point(blockSize_x * (i + 1), blockSize_y * (j + 1)), GridColor, lineWidth);
				putText(showFrame, to_string(quadBins[i][j]), Point((blockSize_x* (i)), (blockSize_y* (j+1))), FONT_HERSHEY_PLAIN, 1, GridColor, 2);
				circle(showFrame, Point(blockSize_x * (i + 0.5), blockSize_y * (j + 0.5)), block_radio, GridColor);
			}
		}
		imshow("Live", showFrame);
		int key;
		if ( (key = waitKey(5)) >= 0) {
			if (key == INTRO_KEY) {
				if (CheckFrame(frame, quadBins, 20, pattern_cols, pattern_rows, blockSize_x, blockSize_y, n_rows, n_columns, checkInvariants)) {
					string name = "image-" + to_string(frame_cont) + ".png";
					imwrite(name,frame);
					name = path + '/' + name;
					#ifdef _WIN32
						replace(name.begin(), name.end(), '/', '\\');
					#else
						replace(name.begin(), name.end(), '\\', '/');
					#endif		
					images  << name << endl;
					frame_cont++;
				}
			}
			else if (key == K_KEY || key == C_KEY) {
				break;
			}
			else if (key == ESC_KEY) {
				break;
				return 0;
			}
			else {
				cout << key << endl;
				cout << "Comando no reconocido" << endl;
			}
		}
	}
	CameraCalibrationRing camera_calibration((path+"/images.txt"), pattern_cols, pattern_rows, square_size);
	camera_calibration.calibrate_camera_iterative(10, n_frames, n_rows, n_columns, actualVid);
	
	
	
	/*cout << "Rotation " << endl;
	for (int i = 0; i < camera_calibration.rvecs.size(); i++)
	{
		cout << camera_calibration.rvecs[i] << endl;
	}
	cout << "Translation " << endl;
	for (int i = 0; i < camera_calibration.tvecs.size(); i++)
	{
		cout << camera_calibration.tvecs[i] << endl;
	}*/
	//WRITE XML TO TEST!
	nextPose(pattern_cols, pattern_rows, square_size, camera_calibration.rvecs,camera_calibration.tvecs, camera_calibration.camera_matrix, camera_calibration.dist_coeffs);
	cout << "fin del programa" << endl;
    return 0;
}
