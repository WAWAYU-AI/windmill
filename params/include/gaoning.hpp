#ifndef _GAONING_HPP
#define _GAONING_HPP
#include <globalParam.hpp>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <random>
#include <stdlib.h>
#include <time.h>
class gaoning
{
private:
    double cnt;

public:
    gaoning();
    void Faker(Eigen::Vector3d &position, double &yaw, double v_yaw, double times, void (*getPoint)(Eigen::Vector3d &, int));
};
void points1(Eigen::Vector3d &position, int cnt);
void points2(Eigen::Vector3d &position, int cnt);
void points3(Eigen::Vector3d &position, int cnt);
#define GRAVITY_G (9.8f)
#define AIR_RAMP_RATIO (0.05f)
#define SHOOT_LENGTH (200.f / 1000.f)
typedef struct
{
    float targetPitchValue;
    float targetYawValue;
    float timeResult;
} myBulletOutput_t;
class myBulletRevise_t
{
public:
    myBulletRevise_t();

public:
    float k;
    float g;
    float lengthActive;

public:
    float speedBulletValue;

public:
    cv::Point3f speedTargetWorld;
    cv::Point3f speedTargetRel;
    cv::Point3f speedThis;
    cv::Point3f posTargetRel;
    struct direct_t
    {
        int8_t xDirect;
        int8_t yDirect;
    } directFlight;

public:
    myBulletOutput_t bulletOutput;

public:
    float myBulletModelRunTime();

protected:
    virtual void getTargetState();
    void thisSpeedGet(const float &vx, const float &vy);
    void myBulletModelRun();

private:
    float flightTime2Yaw(float timeIter) const;
    float flightTime2Pitch(float timeIter) const;
    float flightTime2DistX(float timeIter);
    float flightTime2DistY(float timeIter);
    void flightTimeIter();
};

class bulletToGaoNing_t : myBulletRevise_t
{
private:
    myBulletRevise_t myBulletRevise;
    cv::Point3f myPoint;

public:
    float getFlightTime(cv::Point3f myPoint, const float &bulletSpeed);
    void getTargetState();
};

void convertNumber(const std::string &number_s,int &number_i);
#endif //_GAONING_HPP