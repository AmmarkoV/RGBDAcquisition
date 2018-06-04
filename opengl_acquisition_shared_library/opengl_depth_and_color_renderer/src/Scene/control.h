#ifndef CONTROL_H_INCLUDED
#define CONTROL_H_INCLUDED


int printObjectData(unsigned int objectToPrint);


int moveObject(unsigned objToMove , float X , float Y , float Z);

int rotateObject(unsigned objToMove , float X , float Y , float Z , float angleDegrees);



int handleUserInput(char key,int state,unsigned int x, unsigned int y);


#endif  // CONTROL_H_INCLUDED
