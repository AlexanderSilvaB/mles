#include "Pong.hpp"
#include <sstream>

using namespace std;
using namespace cv;

Pong::Pong(int width, int height)
{
    img = Mat::zeros(height, width, CV_8UC1);

    speedIncr = 0.001f;

    int w_2 = width / 2;
    int h_2 = height / 2;

    wP = width / 50.0f;
    hP = height / 12.0f;
    bR = height / 26.0f;

    ballX = w_2;
    ballY = h_2;
    p1X = 2*wP;
    p1Y = h_2;
    p2X = width - 2*wP;
    p2Y = h_2;

    baseV = 100;

    p1V = 0;
    p2V = 0;
    bVx = -1;
    bVy = bVy = (rand() % 100) > 50 ? 1 : -1;

    scoreP1 = 0;
    scoreP2 = 0;

    count = 0;

    key = 0;
    mode = BEST;

    finished = false;
    ready = false;
}

Pong::~Pong()
{

}

void Pong::setMode(PongModes mode)
{
    this->mode = mode;
}

float Pong::getPlayerSpeed()
{
    return p2V;
}

void Pong::setPlayerSpeed(float v)
{
    p2V = v;
}

void Pong::getBallPosition(float& x, float& y)
{
    x = ballX;
    y = ballY;
}

void Pong::getPlayerPosition(float& y)
{
    y = p2Y;
}

void Pong::getOponentPosition(float& y)
{
    y = p1Y;
}

float Pong::getBallSpeed()
{
    return baseV;
}

void Pong::setBallSpeedIncr(float speedIncr)
{
    this->speedIncr = speedIncr;
}

void Pong::getScore(int& p1, int& p2)
{
    p1 = scoreP1;
    p2 = scoreP2;
}

bool Pong::isReady()
{
    return ready;
}

bool Pong::isFinished()
{
    return finished;
}

void Pong::update()
{
    img.setTo(Scalar(0));

    if(finished)
    {
        string text = "Finished!";
        int baseline = 0;
        int thickness = 1;
        double fontScale = 4.0;
        int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
        Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
        baseline += thickness;

        // center the text
        Point textOrg((img.cols - textSize.width)/2, (img.rows + textSize.height)/2);

        // draw the box
        rectangle(img, textOrg + Point(0, baseline), textOrg + Point(textSize.width, -textSize.height), Scalar(255));
        // ... and the baseline first
        line(img, textOrg + Point(0, textSize.height / 3), textOrg + Point(textSize.width, textSize.height / 3), Scalar(255));

        // then put the text itself
        putText(img, text, textOrg + Point(0, textSize.height / 4), fontFace, fontScale, Scalar(255), thickness, CV_AA);

        imshow("Pong", img);
        waitKey(30);
        return;
    }

    if(!ready)
    {
        string text = "PONG";
        int baseline = 0;
        int thickness = 1;
        double fontScale = 4.0;
        int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
        Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
        baseline += thickness;

        // center the text
        Point textOrg((img.cols - textSize.width)/2, (img.rows + textSize.height)/2);

        // draw the box
        rectangle(img, textOrg + Point(0, baseline), textOrg + Point(textSize.width, -textSize.height), Scalar(255));
        // ... and the baseline first
        line(img, textOrg + Point(0, textSize.height / 3), textOrg + Point(textSize.width, textSize.height / 3), Scalar(255));

        // then put the text itself
        putText(img, text, textOrg + Point(0, textSize.height / 4), fontFace, fontScale, Scalar(255), thickness, CV_AA);


        imshow("Pong", img);
        waitKey(30);
        count++;
        if(count > 100)
        {
            count = 0;
            ready = true;
            img.setTo(Scalar(0));
        }
        return;
    }

    baseV += count * speedIncr;

    ballX += baseV * bVx * 0.03f;
    ballY += baseV * bVy * 0.03f;

    p1Y += p1V * 0.03f;
    p2Y += p2V * 0.03f;
    if(p1Y < hP)
        p1Y = hP;
    else if(p1Y > img.rows - hP)
        p1Y = img.rows - hP;
    if(p2Y < hP)
        p2Y = hP;
    else if(p2Y > img.rows - hP)
        p2Y = img.rows - hP;

    if(ballX <= bR)
    {
        scoreP2++;
        ballY = p2Y = p1Y = img.rows / 2;
        ballX = p2X - 2*wP;
        bVx = -1;
        bVy = (rand() % 100) > 50 ? 1 : -1;
        baseV = 100;
        count = 0;
        p2V = 0;
    }
    else if(ballX >= img.cols - bR)
    {
        scoreP1++;
        ballY = p2Y = p1Y = img.rows / 2;
        ballX = p1X + 2*wP;
        bVx = 1;
        bVy = (rand() % 100) > 50 ? 1 : -1;
        baseV = 100;
        count = 0;
        p2V = 0;
    }

    if(ballY <= bR || ballY >= img.rows - bR)
        bVy *= -1;

    if(ballX <= p1X + 2*wP && ballY >= p1Y - hP && ballY <= p1Y + hP && bVx < 0)
        bVx *= -1;

    if(ballX >= p2X - 2*wP && ballY >= p2Y - hP && ballY <= p2Y + hP && bVx > 0)
        bVx *= -1;

    p1V = (20.0f + ( (rand() % 1000) / 200.0f) ) * (ballY - p1Y);

    if(mode == BEST)
        p2V = (30.0f + ( (rand() % 1000) / 200.0f) ) * (ballY - p2Y);
    else if(mode == GOOD)
        p2V = (20.0f + ( (rand() % 1000) / 200.0f) ) * (ballY - p2Y);
    else if(mode == MANUAL)
    {
        p2V = p2V * 0.99f;
        if(key == 'w')
        {
            if(p2V > 0)
                p2V = -100.0f;
            else
                p2V -= 20.0f;
        }
        else if(key == 's')
        {
            if(p2V < 0)
                p2V = 100.0f;
            else
                p2V += 20.0f;
        }
    }

    line(img, Point(img.cols/2, 0), Point(img.cols/2, img.rows), Scalar(255), 1);
    circle(img, Point(ballX, ballY), bR, Scalar(255), CV_FILLED);
    rectangle(img, Rect(p1X - wP, p1Y - hP, 2*wP, 2*hP), Scalar(255), CV_FILLED);
    rectangle(img, Rect(p2X - wP, p2Y - hP, 2*wP, 2*hP), Scalar(255), CV_FILLED);

    stringstream ss;
    ss << scoreP1;
    putText(img, 
            ss.str(),
            Point(img.cols/2 - 4*wP, 1*hP), // Coordinates
            FONT_HERSHEY_COMPLEX_SMALL, // Font
            1.0, // Scale. 2.0 = 2x bigger
            Scalar(255), // BGR Color
            1, // Line Thickness (Optional)
            CV_AA);

    ss.str("");
    ss << scoreP2;
    putText(img, 
            ss.str(),
            Point(img.cols/2 + 3*wP, 1*hP), // Coordinates
            FONT_HERSHEY_COMPLEX_SMALL, // Font
            1.0, // Scale. 2.0 = 2x bigger
            Scalar(255), // BGR Color
            1, // Line Thickness (Optional)
            CV_AA);

    imshow("Pong", img);
    key = waitKey(30);
    count++;
}

void Pong::reset()
{
    ballX = img.cols / 2;
    ballY = img.rows / 2;
    p1Y = ballY;
    p2Y = ballY;

    baseV = 100;

    p1V = 0;
    p2V = 0;
    bVx = -1;
    bVy = bVy = (rand() % 100) > 50 ? 1 : -1;

    scoreP1 = 0;
    scoreP2 = 0;

    count = 0;

    key = 0;

    ready = false;
    finished = false;
}

void Pong::finish()
{
    finished = true;
    update();
}