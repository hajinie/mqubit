#include <iostream> // 표준 헤더 파일에서는 확장자 생략

int main(void)
{
    int num = 20;
    std::cout << "Hello World!" << std::endl;  // std::cout은 C언어에서의 스트림 stdout에 해당 
    std::cout << "Hello " << "World!" << std::endl;  // std::endl은 C언어에서의 '\n'에 해당 
    std::cout << num << ' ' << 'A';
    std::cout << ' ' << 3.14 << std::endl;

    system("pause");  // VC++ 에서만 필요
    return 0;
}
