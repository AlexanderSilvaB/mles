#include "Snake.hpp"
#include <sstream>

using namespace std;
using namespace cv;

Snake::Snake(int width, int height, int size)
{
    lives = 3;
    dt = 30;
    mode = MANUAL;
    score = 0;
    key = 0;
    count = 0;
    command = STRAIGHT;
    direction = 0;
    this->width = 0;
    this->height = 0;
    this->size = 0;
    resize(width, height, size);
}

Snake::~Snake()
{

}

void Snake::resize(int width, int height, int size)
{
    if(width == this->width && height == this->height && size == this->size)
    {
        return;
    }
    this->width = width;
    this->height = height;
    this->size = size;

    textLimit = height / 10;

    img = Mat::zeros(height*size, width*size, CV_8UC1);
    
    headX = width / 2;
    headY = height / 2;
    lastHeadX = headX;
    lastHeadY = headY;

    generateBerry();

    finished = false;
    ready = false;
}

void Snake::setMode(SnakeModes mode)
{
    this->mode = mode;
}

void Snake::control(SnakeCommands command)
{
    this->command = command;
}

void Snake::getHeadPosition(int& x, int& y)
{
    x = headX;
    y = headY;
}

void Snake::getBerryPosition(int& x, int& y)
{
    x = berryX;
    y = berryY;
}

int Snake::getScore()
{
    return score;
}

int Snake::getWidth()
{
    return width;
}

int Snake::getHeight()
{
    return height;
}

bool Snake::isReady()
{
    return ready;
}

bool Snake::isFinished()
{
    return finished;
}

void Snake::setInterval(int dt)
{
    this->dt = dt;
}

int Snake::getDirection()
{
    return direction;
}

int Snake::getFoodDirection()
{
    if(berryY < headY)
        return 0;
    else if(berryX > headX)
        return 1;
    else if(berryY > headY)
        return 2;
    else if(berryX < headX)
        return 3;
    return 0;
}

void Snake::getDanger(bool& left, bool& top, bool& right, bool& bottom)
{
    left = headX == 1;
    right = headX == (width - 2);
    top = headY == textLimit+1;
    bottom = headY == (height - 2);

    for(int i = 0; i < tail.size(); i++)
    {
        left |= headX == (tail[i].first + 1) && headY == tail[i].second;
        right |= headX == (tail[i].first - 1) && headY == tail[i].second;
        top |= headX == tail[i].first && headY == (tail[i].second + 1);
        bottom |= headX == tail[i].first && headY == (tail[i].second - 1);
    }
}

void Snake::generateBerry()
{
    berryX = 1 + ( rand() % (width - 2) );
    berryY = (textLimit + 1) + ( rand() % (height - textLimit - 2) );
}

int Snake::update()
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

        imshow("Snake", img);
        if(dt > 0)
            waitKey(30);
        return 0;
    }

    if(!ready)
    {
        string text = "Snake";
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


        imshow("Snake", img);
        if(dt > 0)
            waitKey(30);
        count++;
        if(count > 100)
        {
            count = 0;
            ready = true;
            img.setTo(Scalar(0));
        }
        return 0;
    }

    if(mode == MANUAL)
    {
        
    }

    switch(command)
    {
        case LEFT:
            direction++;
            if(direction > 3)
                direction = 0;
            break;
        case RIGHT:
            direction--;
            if(direction < 0)
                direction = 3;
            break;
        default:
            break;
    }

    lastHeadX = headX;
    lastHeadY = headY;
    switch(direction)
    {
        case 0:
            headY--;
            break;
        case 1:
            headX++;
            break;
        case 2:
            headY++;
            break;
        case 3:
            headX--;
            break;
        default:
            break;
    }

    bool hit = false;
    bool eat = false;
    if(headX < 1)
    {
        hit = true;
    }
    else if(headX >= width-1)
    {
        hit = true;
    }
    if(headY <= textLimit)
    {
        hit = true;
    }
    else if(headY >= height-1)
    {
        hit = true;
    }
    
    if(!hit)
    {
        for(int i = 0; i < tail.size(); i++)
        {
            hit = headX == tail[i].first && headY == tail[i].second;
            if(hit)
                break;
        }
    }

    if(headX == berryX && headY == berryY)
    {
        eat = true;
    }

    pair<int, int> tailEnd = pair<int, int>(lastHeadX, lastHeadY);
    if(tail.size() > 0)
    {
        tailEnd = tail.back();
        for(int i = tail.size()-1; i > 0; i--)
            tail[i] = tail[i-1];
        tail[0] = pair<int, int>(lastHeadX, lastHeadY);
    }

    line(img, Point(0, size*textLimit + size*0.5), Point(img.cols, size*textLimit + size*0.5), Scalar(150), size);
    line(img, Point(img.cols-size*0.5, size*textLimit + size*0.5), Point(img.cols-size*0.5, img.rows-size*0.5), Scalar(150), size);
    line(img, Point(img.cols, img.rows-size*0.5), Point(size, img.rows-size*0.5), Scalar(150), size);
    line(img, Point(size*0.5, img.rows-size*0.5), Point(size*0.5, size*textLimit + size*0.5), Scalar(150), size);

    double r = size*0.5;
    circle(img, Point(berryX*size + r, berryY*size + r), r, Scalar(255), CV_FILLED);
    rectangle(img, Rect(headX*size, headY*size, size, size), Scalar(255), CV_FILLED);


    for(int i = 0; i < tail.size(); i++)
    {
        rectangle(img, Rect(tail[i].first*size, tail[i].second*size, size, size), Scalar(220), CV_FILLED);
    }

    stringstream ss;
    ss << "Lives: " << lives << "   Score: " << score;
    putText(img, 
            ss.str(),
            Point(size, 3*size), // Coordinates
            FONT_HERSHEY_COMPLEX_SMALL, // Font
            1.0, // Scale. 2.0 = 2x bigger
            Scalar(255), // BGR Color
            1, // Line Thickness (Optional)
            CV_AA);

    imshow("Snake", img);
    if(dt > 0)
        key = waitKey(dt);
    count++;

    if(hit)
    {
        lives--;
        tail.clear();
        headX = width / 2;
        headY = height / 2;
        lastHeadX = headX;
        lastHeadY = headY;
        generateBerry();
        direction = 0;
        if(lives < 0)
            finish();
        return -1;
    }
    if(eat)
    {
        score++;
        tail.push_back(tailEnd);
        generateBerry();
        return 1;
    }

    return 0;
}

void Snake::reset()
{
    tail.clear();
    headX = width / 2;
    headY = height / 2;
    lastHeadX = headX;
    lastHeadY = headY;
    generateBerry();

    score = 0;

    count = 0;

    lives = 3;

    key = 0;

    ready = false;
    finished = false;
}

void Snake::finish()
{
    finished = true;
    update();
}