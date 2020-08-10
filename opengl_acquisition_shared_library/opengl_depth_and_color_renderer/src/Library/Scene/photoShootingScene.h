#ifndef PHOTOSHOOTINGSCENE_H_INCLUDED
#define PHOTOSHOOTINGSCENE_H_INCLUDED



int setupPhotoshoot(
                        void * context,
                        int objID,
                        unsigned int columns , unsigned int rows ,
                        float distance,
                        float angleX,float angleY,float angleZ ,
                        float angXVariance ,float angYVariance , float angZVariance
                       );

void * createPhotoshoot(
                        void * scene,
                        void * modelStorage,
                        int objID,
                        unsigned int columns , unsigned int rows ,
                        float distance,
                        float angleX,float angleY,float angleZ ,
                        float angXVariance ,float angYVariance , float angZVariance
                       );

int renderPhotoshoot( void * context  );
int sceneSwitchKeyboardControl(int newVal);


#endif //PHOTOSHOOTINGSCENE_H_INCLUDED
