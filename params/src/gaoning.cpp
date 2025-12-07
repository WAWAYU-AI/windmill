#include "gaoning.hpp"
gaoning::gaoning()
{
    this->cnt = 0;
}
void expDecay(float a, float &b)
{
    b = 1 - exp(-a);
}
void points1(Eigen::Vector3d &position, int cnt)
{
    int now_cnt = cnt % 1000;
    if (now_cnt <= 250 && now_cnt >= 0)
    {
        position(0) = (double)now_cnt / 100;
        position(1) = 0;
    }
    else if (now_cnt <= 500 && now_cnt > 250)
    {
        position(0) = 2.5;
        position(1) = (double)(now_cnt - 250) / 100;
    }
    else if (now_cnt <= 750 && now_cnt > 500)
    {
        position(0) = (double)(750 - now_cnt) / 100;
        position(1) = 2.5;
    }
    else if (now_cnt <= 1000 && now_cnt > 750)
    {
        position(0) = 0;
        position(1) = (double)(1000 - now_cnt) / 100;
    }
}
void points2(Eigen::Vector3d &position, int cnt)
{
    int now_cnt = fmod(cnt, 628);
    position(0) = 0.5 * cos((double)now_cnt / 100);
    position(1) = 0.5 * sin((double)now_cnt / 100);
}
void points3(Eigen::Vector3d &position, int cnt)
{
    int now_cnt = cnt % 400;
    if (now_cnt >= 0 && now_cnt < 100)
    {
        position(0) = 0.05 * now_cnt;
        position(1) = 0;
    }
    else if (now_cnt >= 100 && now_cnt < 200)
    {
        position(0) = 5;
        position(1) = 0;
    }
    else if (now_cnt >= 200 && now_cnt < 300)
    {
        position(0) = 5 - 0.05 * (now_cnt - 200);
        position(1) = 0;
    }
    else if (now_cnt >= 300 && now_cnt < 400)
    {
        position(0) = 0;
        position(1) = 0;
    }
}
void gaoning::Faker(Eigen::Vector3d &position, double &yaw, double v_yaw, double times, void (*getPoint)(Eigen::Vector3d &, int))
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> noise(0, 0.000001);
    (*getPoint)(position, cnt);
    position += Eigen::Vector3d(noise(gen), noise(gen), 0);
    cv::Point2f center(position(0), position(1));
    yaw = fmod(v_yaw * cnt * 0.02, 1.57) - 0.78;
    yaw += noise(gen) * 0.02;
    position(0) -= 0.28 * cos(yaw);
    position(1) += 0.28 * sin(yaw);
    cv::Point2f armor(position(0), position(1));
    cv::Mat image = cv::Mat::zeros(960, 1280, CV_8UC3);
    image.setTo(cv::Scalar(211, 211, 211));
    cv::circle(image, cv::Point2f(480, 640) - (center * 1000 * times), 5, cv::Scalar(193, 182, 255), -1, 8, 0);
    cv::circle(image, cv::Point2f(480, 640) - (armor * 1000 * times), 2, cv::Scalar(0, 0, 255), -1, 8, 0);
    cv::line(image, cv::Point2f(480, 640) - (center * 1000 * times), cv::Point2f(480, 640) - (armor * 1000 * times), cv::Scalar(255, 0, 0), 1);
    cv::circle(image, cv::Point2f(480, 640) - ((2 * center - armor) * 1000 * times), 2, cv::Scalar(0, 0, 255), -1, 8, 0);
    cv::line(image, cv::Point2f(480, 640) - (center * 1000 * times), cv::Point2f(480, 640) - ((2 * center - armor) * 1000 * times), cv::Scalar(0, 0, 255), 1);
    cv::circle(image, cv::Point2f(480, 640) - ((center - cv::Point2f(center.y - armor.y, -center.x + armor.x)) * 1000 * times), 2, cv::Scalar(0, 0, 255), -1, 8, 0);
    cv::line(image, cv::Point2f(480, 640) - (center * 1000 * times), cv::Point2f(480, 640) - ((center - cv::Point2f(center.y - armor.y, -center.x + armor.x)) * 1000 * times), cv::Scalar(0, 0, 255), 1);
    cv::circle(image, cv::Point2f(480, 640) - ((center + cv::Point2f(center.y - armor.y, -center.x + armor.x)) * 1000 * times), 2, cv::Scalar(0, 0, 255), -1, 8, 0);
    cv::line(image, cv::Point2f(480, 640) - (center * 1000 * times), cv::Point2f(480, 640) - ((center + cv::Point2f(center.y - armor.y, -center.x + armor.x)) * 1000 * times), cv::Scalar(0, 0, 255), 1);
#ifdef DEBUGMODE
    cv::imshow("real state", image);
#endif // DEBUGMODE
    cnt++;
    return;
}
myBulletRevise_t::myBulletRevise_t() : g(GRAVITY_G), k(AIR_RAMP_RATIO), lengthActive(SHOOT_LENGTH)
{
    this->speedBulletValue = 13.3f;
}

float myBulletRevise_t::flightTime2Yaw(float timeIter) const
{
    float decayEvent = 0.f;
    expDecay(k * timeIter, decayEvent);
    float yawUp = k * (posTargetRel.x + timeIter * speedTargetWorld.x) - speedThis.x * decayEvent;
    float yawDown = k * (posTargetRel.y + timeIter * speedTargetWorld.y) - speedThis.x * decayEvent;
    //return atan(yawUp/yawDown)*180.f/PI;
    return atan2f(yawUp, yawDown) * 180.f / M_PI;
}
//��Ŀ�����ʱ�䵽Ŀ��Pitch��Ƕ�
float myBulletRevise_t::flightTime2Pitch(float timeIter) const
{
    float decayEvent = 0.f;
    expDecay(k * timeIter, decayEvent);
    float pitchUp = k * k * posTargetRel.z - g * decayEvent + g * timeIter * k;
    float pitchDown = lengthActive * k * k + speedBulletValue * decayEvent * k;
    return asinf(pitchUp / pitchDown) * 180.f / M_PI;
}
float myBulletRevise_t::flightTime2DistX(float timeIter)
{
    float decayEvent = 0.f;
    expDecay(k * timeIter, decayEvent);
    float pitchArc = flightTime2Pitch(timeIter) * M_PI / 180.f;                                            //������Pitch�Ƕ�
    float yawArc = flightTime2Yaw(timeIter) * M_PI / 180.f;                                                //������Yaw��
    float eventBullet = (speedThis.x + speedBulletValue * sinf(yawArc) * cosf(pitchArc)) * decayEvent / k; //�ӵ��ƶ���
    float eventChase = speedTargetWorld.x * timeIter;                                                      //Ŀ���ƶ���
    return eventBullet - eventChase;
}

//��Ŀ�����ʱ�䵽Ŀ�����Y
float myBulletRevise_t::flightTime2DistY(float timeIter)
{
    float decayEvent = 0.f;
    expDecay(k * timeIter, decayEvent);
    float pitchArc = flightTime2Pitch(timeIter) * M_PI / 180.f;                                            //������Pitch�Ƕ�
    float yawArc = flightTime2Yaw(timeIter) * M_PI / 180.f;                                                //������Yaw��
    float eventBullet = (speedThis.y + speedBulletValue * cosf(yawArc) * cosf(pitchArc)) * decayEvent / k; //�ӵ��ƶ���
    float eventChase = speedTargetWorld.y * timeIter;                                                      //Ŀ���ƶ���
    return eventBullet - eventChase;
}
//���õ��������з���ʱ�����
void myBulletRevise_t::flightTimeIter()
{
    //x,y�������ĸ����볤,����һ��������е���
    float (myBulletRevise_t::*iterDist)(float) = &myBulletRevise_t::flightTime2DistY;
    float iterInitDist = sqrt(posTargetRel.x * posTargetRel.x + posTargetRel.y * posTargetRel.y);
    float *ptrTargetPos = &posTargetRel.y;
    if (fabs(posTargetRel.x) >= fabs(posTargetRel.y))
    {
        iterDist = &myBulletRevise_t::flightTime2DistX;
        ptrTargetPos = &posTargetRel.x;
    }
    float timeFirst = fabs(iterInitDist / speedBulletValue);
    float timeLast = 1.2f * timeFirst;
    float errorDist = *ptrTargetPos - (this->*iterDist)(timeLast);
    for (int i = 0; i < 10; i++)
    {
        if (fabs(errorDist) <= 0.001)
        {
            break;
        }
        float timeBuffer = timeLast - (-errorDist) * (timeLast - timeFirst) / ((this->*iterDist)(timeLast) - (this->*iterDist)(timeFirst));
        timeFirst = timeLast;
        timeLast = timeBuffer;
        errorDist = *ptrTargetPos - (this->*iterDist)(timeLast);
    }
    bulletOutput.timeResult = timeLast;
    bulletOutput.targetYawValue = flightTime2Yaw(bulletOutput.timeResult);
    bulletOutput.targetPitchValue = flightTime2Pitch(bulletOutput.timeResult);
    static float pitchLast = 0.f, yawLast = 0.f;
    if (bulletOutput.targetYawValue != bulletOutput.targetYawValue)
    {
        bulletOutput.targetYawValue = yawLast;
    }
    else
    {
        yawLast = bulletOutput.targetYawValue;
    }
    if (bulletOutput.targetPitchValue != bulletOutput.targetPitchValue)
    {
        bulletOutput.targetPitchValue = pitchLast;
    }
    else
    {
        pitchLast = bulletOutput.targetPitchValue;
    }
}
void myBulletRevise_t::thisSpeedGet(const float &vx, const float &vy)
{
    this->speedThis.x = vx;
    this->speedThis.y = vy;
}
void myBulletRevise_t::getTargetState()
{
    speedTargetRel.x = 0.f;
    speedTargetRel.y = 0.f;
    speedTargetRel.z = 0.f;
    speedTargetWorld = this->speedTargetRel + this->speedThis;
}
void myBulletRevise_t::myBulletModelRun()
{
    speedThis = cv::Point3f(0, 0, 0);
    speedTargetWorld = speedTargetRel + speedThis;
    directFlight.xDirect = (posTargetRel.x >= 0.f) ? 1 : (-1);
    directFlight.yDirect = (posTargetRel.y >= 0.f) ? 1 : (-1);
    getTargetState();
    flightTimeIter();
}

float myBulletRevise_t::myBulletModelRunTime()
{
    myBulletModelRun();
    return bulletOutput.timeResult;
}

void bulletToGaoNing_t::getTargetState()
{
    posTargetRel.x = -myPoint.y;
    posTargetRel.y = myPoint.x;
    posTargetRel.z = myPoint.z;
    speedTargetRel = cv::Point3f(0, 0, 0);
}

float bulletToGaoNing_t::getFlightTime(cv::Point3f myPoint, const float &bulletSpeed)
{
    this->myPoint = myPoint;
    speedBulletValue = bulletSpeed;
    getTargetState();
    return this->myBulletModelRunTime();
}

void convertNumber(const std::string &number_s, int &number_i)
{
    if (number_s == "outpost")
    {
        number_i = 0;
    }
    else if (number_s == "guard")
    {
        number_i = 6;
    }
    else if (number_s == "base")
    {
        number_i = 7;
    }
    else if (number_s == "1" || number_s == "2" || number_s == "3" || number_s == "4" || number_s == "5")
    {
        number_i = number_s[number_s.size() - 1] - '0';
    }
    else
    {
        // continue;
        number_i = 3;
    }
}
