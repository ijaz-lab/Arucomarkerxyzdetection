#include <iostream>
// Opencv Includes 
#include "opencv2\core.hpp"
#include "opencv2\aruco.hpp"
#include "opencv2\imgcodecs.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\calib3d.hpp"
#include<iterator> 
#include <sstream>
#include <fstream>
using namespace std;
using namespace cv;
int cameraID= 0;
// camera ID
int countImage =0;
string clibrationPrameterFiles ="C:/arucomarkerwithxyzdetection/Files";
const float calibrationSqureDimension = 0.01405f; // meters 
const float arucoSquareDimension = 0.191f; //meters
const Size chessboardDimension = Size(9, 7);
// creating fucntion 

void createKnowBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{    for (int i = 0; i < boardSize.height; i++) {

        for (int j = 0; j < boardSize.width; j++)
        {
            corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
             }
    }
}
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false) {

    for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
    {
        vector<Point2f> pointBuf;
        
       // finding chess board corners

        bool found = findChessboardCorners(*iter, Size(9, 7), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK );
        if (found) {
            allFoundCorners.push_back(pointBuf);

        }
        if (showResults) {
            drawChessboardCorners(*iter, Size(9, 7), pointBuf, found);
            imshow("Looking for Chess Board", *iter);
            waitKey(0);
       }
    }
}
void cameraCalibration(vector<Mat> caliberationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficient) {
    
    vector<vector<Point2f>> checkerboardImagesSpacePoints;

    getChessboardCorners(caliberationImages, checkerboardImagesSpacePoints, false);

    vector<vector<Point3f>> wordSpaceCornersPoints(1);

    createKnowBoardPosition(boardSize, squareEdgeLength, wordSpaceCornersPoints[0]);
    wordSpaceCornersPoints.resize(checkerboardImagesSpacePoints.size(), wordSpaceCornersPoints[0]);

    vector<Mat> rVectors, tVectors;
    distanceCoefficient = Mat::zeros(8, 1 ,CV_64F);
    calibrateCamera(wordSpaceCornersPoints, checkerboardImagesSpacePoints, boardSize, cameraMatrix, distanceCoefficient, rVectors, tVectors);

}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
    ofstream outStream(name);
    if (outStream)
    {
        uint16_t rows = cameraMatrix.rows;
        uint16_t columns = cameraMatrix.cols;

        outStream << rows << endl;
        outStream << columns << endl;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                double value = cameraMatrix.at<double>(r, c);
                outStream << value << endl;
            }
        }

        rows = distanceCoefficients.rows;
        columns = distanceCoefficients.cols;

        outStream << rows << endl;
        outStream << columns << endl;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                double value = distanceCoefficients.at<double>(r, c);
                outStream << value << endl;
            }
        }
        outStream.close();
        return true;
    }
    return false;
}


bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
    ifstream inStream(name);
    if (inStream)
    {
        uint16_t rows;
        uint16_t columns;

        inStream >> rows;
        inStream >> columns;

        cameraMatrix = Mat(Size(columns, rows), CV_64F);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                double read = 0.0f;
                inStream >> read;
                cameraMatrix.at<double>(r, c) = read;
                cout << cameraMatrix.at<double>(r, c) << "\n";
            }
        }
        //Disctance coefficients
        inStream >> rows;
        inStream >> columns;
        //cout << rows<< endl;

        distanceCoefficients = Mat::zeros(rows, columns, CV_64F);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                double read = 0.0f;
                inStream >> read;
                distanceCoefficients.at<double>(r, c) = read;
                cout << distanceCoefficients.at<double>(r, c) << "\n";
            }
        }
        inStream.close();
        return true;
    }
    return false;
}

int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimensions)
{
    Mat frame;
    cout << "camera" << cameraMatrix;
    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners, rejectedCandidates;
    aruco::DetectorParameters parameters;

    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_250);

    VideoCapture vid(cameraID);
    //VideoCapture vid(1);
    if (vid.isOpened() == false)  //  To check if object was associated to webcam successfully
    {
        std::cout << "error: Webcam connect unsuccessful\n"; // if not then print error message
        return(0);            // and exit program
    }
    /*if (!vid.isOpened())
    {
        return -1;
    }*/

    namedWindow("Webcam", WINDOW_AUTOSIZE);

    vector<Vec3d> rotationVectors, translationVectors;

    while (true)
    {
        if (!vid.read(frame))
            break;

        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix,
            distanceCoefficients, rotationVectors, translationVectors);

        for (int i = 0; i < markerIds.size(); i++)
        {
            aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.1f);
            cout << "detection" << endl;
        }
        imshow("Webcam", frame);
        if (waitKey(30) >= 0) break;
    }
    return 1;
}
void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients)
{
    Mat frame;
    Mat drawToFrame;


    vector<Mat> savedImages;

    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    VideoCapture vid(cameraID);
    //VideoCapture vid(1);
    if (vid.isOpened() == false)  //  To check if object was associated to webcam successfully
    {
        std::cout << "error: Webcam connect unsuccessful\n"; // if not then print error message
        return;            // and exit program
    }
    /*if (!vid.isOpened())
    {
        return;
    }*/

    int framesPerSecond = 20;

    namedWindow("Webcam", WINDOW_NORMAL);

    while (true)
    {
        if (!vid.read(frame))
            break;

        vector<Vec2f> foundPoints;
        bool found = false;

        found = findChessboardCorners(frame, chessboardDimension, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        frame.copyTo(drawToFrame);
        drawChessboardCorners(drawToFrame, chessboardDimension, foundPoints, found);
        if (found)
            imshow("Webcam", drawToFrame);
        else
            imshow("Webcam", frame);
        char character = waitKey(1000/ 30);

        switch (character)
        {
        case ' ':
            //saving image
            if (found)
            {
                Mat temp;
                frame.copyTo(temp);
                savedImages.push_back(temp);
                // imwrite("./image.png",frame);
                cout<<"Saving Image"<<endl;
            }
            break;
        case 13:
            //start calibration
            if (savedImages.size() > 15)
            {
                cameraCalibration(savedImages, chessboardDimension, calibrationSqureDimension, cameraMatrix, distanceCoefficients);
                saveCameraCalibration(clibrationPrameterFiles, cameraMatrix, distanceCoefficients);
                cout<<"Calibrating"<<endl;
            }
            break;
        case 27:

            //exit
        return;
            break;
          }
       }
     }
     int main(int, char**) {
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distanceCoefficients;
    // cameraCalibrationProcess(cameraMatrix, distanceCoefficients);
    // startWebcamMonitoring(cameraMatrix, distanceCoefficients, 0.099f);
      loadCameraCalibration(clibrationPrameterFiles, cameraMatrix, distanceCoefficients);
      startWebcamMonitoring(cameraMatrix, distanceCoefficients, arucoSquareDimension);
    return 0;   
   
}

