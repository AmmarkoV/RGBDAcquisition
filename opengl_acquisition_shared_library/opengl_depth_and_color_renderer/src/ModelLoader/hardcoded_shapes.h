#ifndef HARDCODED_SHAPES_H_INCLUDED
#define HARDCODED_SHAPES_H_INCLUDED



#define TEXHALF 0.5
#define U 0.5
#define PLANE 108.0


static const float planeCoords[]={ //X  Y  Z       W
                     //Bottom
                      U*PLANE, 0,  U*PLANE, //1.0,  // top right
                     -U*PLANE, 0,  U*PLANE, //1.0,  // top left
                     -U*PLANE, 0, -U*PLANE, //1.0,  // bottom left

                      U*PLANE, 0, -U*PLANE,  //, 1.0 // bottom right
                      U*PLANE, 0,  U*PLANE,//1.0,   // top right
                     -U*PLANE, 0, -U*PLANE //1.0,  // bottom left
                  };

static const float planeNormals[]={ //X  Y  Z       W
                      //Bottom
                      0.0f, 1.0f , 0.0f ,// 1.0,
                      0.0f, 1.0f , 0.0f //, 1.0
                  };

static const float planeTexCoords[]={ //X  Y  Z       W
                        //Bottom
                         1.0f, 1.0f, // top right
                         0.0f, 1.0f, // top left
                         0.0f, 0.0f, // bottom left

                         1.0f, 0.0f,  // bottom right
                         1.0f, 1.0f, // top right
                         0.0f, 0.0f // bottom left
                     };

	// The fullscreen quad's FBO
static const float g_quad_vertex_buffer_data[] =
    {
		-1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 1.0f,  1.0f, 0.0f,
	};


static const float cubeCoords[]=
{
-U,-U,-U,
-U,-U, U,
-U, U, U,
 U, U,-U,
-U,-U,-U,
-U, U,-U,
 U,-U, U,
-U,-U,-U,
 U,-U,-U,
 U, U,-U,
 U,-U,-U,
-U,-U,-U,
-U,-U,-U,
-U, U, U,
-U, U,-U,
 U,-U, U,
-U,-U, U,
-U,-U,-U,
-U, U, U,
-U,-U, U,
 U,-U, U,
 U, U, U,
 U,-U,-U,
 U, U,-U,
 U,-U,-U,
 U, U, U,
 U,-U, U,
 U, U, U,
 U, U,-U,
-U, U,-U,
 U, U, U,
-U, U,-U,
-U, U, U,
 U, U, U,
-U, U, U,
 U,-U, U
 };


static const float cubeNormals[]={ //X  Y  Z  W
                      -1.0f,-0.0f,-0.0f,
                      -1.0f,-0.0f,-0.0f,
                      -1.0f,-0.0f,-0.0f,
                       0.0f,0.0f,-1.0f,
                       0.0f,0.0f,-1.0f,
                       0.0f,0.0f,-1.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,0.0f,-1.0f,
                       0.0f,0.0f,-1.0f,
                       0.0f,0.0f,-1.0f,
                      -1.0f,-0.0f,0.0f,
                      -1.0f,-0.0f,0.0f,
                      -1.0f,-0.0f,0.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,-1.0f,0.0f,
                       0.0f,-0.0f,1.0f,
                       0.0f,-0.0f,1.0f,
                       0.0f,-0.0f,1.0f,
                       1.0f,0.0f,-0.0f,
                       1.0f,0.0f,-0.0f,
                       1.0f,0.0f,-0.0f,
                       1.0f,-0.0f,0.0f,
                       1.0f,-0.0f,0.0f,
                       1.0f,-0.0f,0.0f,
                       0.0f,1.0f,0.0f,
                       0.0f,1.0f,0.0f,
                       0.0f,1.0f,0.0f,
                      -0.0f,1.0f,0.0f,
                      -0.0f,1.0f,0.0f,
                      -0.0f,1.0f,0.0f,
                       0.0f,-0.0f,1.0f,
                       0.0f,-0.0f,1.0f,
                       0.0f,-0.0f,1.0f
};

static const float cubeTexCoords[]={ //X  Y  Z       W
                        //Bottom
                         0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 0.0f,
                         0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 0.0f,
                         0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 0.0f,
                         0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 0.0f,
                         0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 0.0f,
                         0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 0.0f,
                         0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 0.0f,
                         0.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 1.0f,
                         1.0f, 1.0f,
                         0.0f, 0.0f,
                         1.0f, 0.0f
                     };



static const float pyramidCoords[]={ //X  Y  Z       W
                     //Far
                     -U, -U, -U,      // bottom left
                      0,  U,  0,      // top
                      U, -U, -U,      // bottom right

                     //Near
                      -U, -U,  U,      // top left
                       U, -U,  U,      // top right
                       0,  U,  0,      // top

                     //Left
                      -U, -U, -U,       // bottom left
                      -U, -U,  U,      // top left
                       0,  U,  0,      // top

                     //Right
                       U, -U, -U,      // bottom right
                       0,  U,  0,      // top
                       U, -U,  U,      // top right


                     //Bottom
                     -U, -U,  U, //1.0,  // top left
                     -U, -U, -U, //1.0,  // bottom left
                      U, -U,  U, //1.0,  // top right

                      U, -U,  U,//1.0,   // top right
                     -U, -U, -U, //1.0,  // bottom left
                      U, -U, -U  //, 1.0 // bottom right
                    };

static const float pyramidNormals[]={ //X  Y  Z  W
                      0.0,0.4472,-0.8944,
                      0.0,0.4472,0.8944,
                      -0.8944,0.4472,0.0,
                      0.8944,0.4472,0.0,
                      0.0,-1.0,0.0,
                      0.0,-1.0,0.0
};

static const float pyramidTexCoords[]={ //X  Y  Z       W
                     //Far
                         0.0f, 0.0f,    // bottom left
                         TEXHALF, 0.0f, // top
                         1.0f, 0.0f,    // bottom right

                     //Near
                         0.0f, 1.0f, // top left
                         1.0f, 1.0f, // top right
                         TEXHALF, 0.0f, // top

                     //Left
                         0.0f, 0.0f, // bottom left
                         0.0f, 1.0f, // top left
                         TEXHALF, 0.0f, // top

                     //Right
                         1.0f, 0.0f,  // bottom right
                         TEXHALF, 0.0f, // top
                         1.0f, 1.0f, // top right

                     //Bottom
                         0.0f, 1.0f, // top left
                         0.0f, 0.0f, // bottom left
                         1.0f, 1.0f, // top right
                         1.0f, 1.0f, // top right
                         0.0f, 0.0f, // bottom left
                         1.0f, 0.0f  // bottom right
                     };


#endif // HARDCODED_SHAPES_H_INCLUDED
