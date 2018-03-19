#pragma once

#include <sys/time.h>
#include <iostream>
#include <ios>     // std::left, std::right
#include <iomanip> // std::setw(int), std::setfill(char)
#include <sstream>
#include <time.h>

class Timer{
public:
  void start();
  void stop();
  float getUsec();
  float getSec();
  float getMin();
  static std::string getCurrentTime(); // edit by kinugawa on 2018/01/10
private:
  struct timeval begin;
  struct timeval end;
};

inline void Timer::start(){
  gettimeofday(&(this->begin), 0);
}

inline void Timer::stop(){
  gettimeofday(&(this->end), 0);
}

inline float Timer::getUsec(){
  return this->end.tv_usec-this->begin.tv_usec;
}

inline float Timer::getSec(){
  return this->end.tv_sec-this->begin.tv_sec;
}

inline float Timer::getMin(){
  return (this->end.tv_sec-this->begin.tv_sec)/60.0;
}

std::string Timer::getCurrentTime(){
  /*
  struct timeval myTime;
  struct tm* time_st;

  // 現在時刻を取得してmyTimeに格納．通常のtime_t構造体とsuseconds_tに値が代入される
  gettimeofday(&myTime, 0);
  time_st = localtime(&(myTime.tv_sec));
*/
  time_t now;
  struct tm* time_st;
  now = time(NULL);
  time_st = localtime(&now);

  std::ostringstream oss;
  oss << time_st->tm_year+1900 << "-";
  oss << std::setfill('0') << std::right << std::setw(2) << time_st->tm_mon+1 << "-";
  oss << std::setfill('0') << std::right << std::setw(2) << time_st->tm_mday << "-"; // day
  oss << std::setfill('0') << std::right << std::setw(2) << time_st->tm_hour << "-"; // hour
  oss << std::setfill('0') << std::right << std::setw(2) << time_st->tm_min; // minutes
  return oss.str();
}
