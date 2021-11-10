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

static unsigned int cubeTriangleCount  =  (unsigned int )  sizeof(cubeCoords)/(3*sizeof(float));

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


// One color for each vertex. They were generated randomly.
static const float cubeColors[] = {
		0.583f,  0.771f,  0.014f,
		0.609f,  0.115f,  0.436f,
		0.327f,  0.483f,  0.844f,
		0.822f,  0.569f,  0.201f,
		0.435f,  0.602f,  0.223f,
		0.310f,  0.747f,  0.185f,
		0.597f,  0.770f,  0.761f,
		0.559f,  0.436f,  0.730f,
		0.359f,  0.583f,  0.152f,
		0.483f,  0.596f,  0.789f,
		0.559f,  0.861f,  0.639f,
		0.195f,  0.548f,  0.859f,
		0.014f,  0.184f,  0.576f,
		0.771f,  0.328f,  0.970f,
		0.406f,  0.615f,  0.116f,
		0.676f,  0.977f,  0.133f,
		0.971f,  0.572f,  0.833f,
		0.140f,  0.616f,  0.489f,
		0.997f,  0.513f,  0.064f,
		0.945f,  0.719f,  0.592f,
		0.543f,  0.021f,  0.978f,
		0.279f,  0.317f,  0.505f,
		0.167f,  0.620f,  0.077f,
		0.347f,  0.857f,  0.137f,
		0.055f,  0.953f,  0.042f,
		0.714f,  0.505f,  0.345f,
		0.783f,  0.290f,  0.734f,
		0.722f,  0.645f,  0.174f,
		0.302f,  0.455f,  0.848f,
		0.225f,  0.587f,  0.040f,
		0.517f,  0.713f,  0.338f,
		0.053f,  0.959f,  0.120f,
		0.393f,  0.621f,  0.362f,
		0.673f,  0.211f,  0.457f,
		0.820f,  0.883f,  0.371f,
		0.982f,  0.099f,  0.879f
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

static unsigned int pyramidTriangleCount  =  (unsigned int )  sizeof(pyramidCoords)/(3*sizeof(float));

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
