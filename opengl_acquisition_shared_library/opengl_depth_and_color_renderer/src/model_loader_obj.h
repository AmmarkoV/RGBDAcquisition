#ifndef MODEL_LOADER_OBJ_H_INCLUDED
#define MODEL_LOADER_OBJ_H_INCLUDED

#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

#define AlmostEqual(v1, v2, tol)                                                    \
	( (fabs(v1.x-v2.x)<tol) && (fabs(v1.y-v2.y)<tol) && (fabs(v1.z-v2.z)<tol) )     \


#define DotProduct(v1, v2) (v1.n1*v2.n1 + v1.n2*v2.n2 + v1.n3*v2.n3)

#define Subtraction(out, v1, v2) {	out.n1 = v1.n1 - v2.n1;			\
									out.n2 = v1.n2 - v2.n2;			\
									out.n3 = v1.n3 - v2.n3;			\
									}

#define VectorLength(v1) (sqrt(v1.n1*v1.n1 + v1.n2*v1.n2 +v1.n3*v1.n3))

#define CrossProduct(out, v1, v2) {		out.n1 = v1.n2*v2.n3 - v1.n3*v2.n2;		\
										out.n2 = v1.n3*v2.n1 - v1.n1*v2.n3;		\
										out.n3 = v1.n1*v2.n2 - v1.n2*v2.n1;		\
										}

#define PI 3.1415936
#define GL_BGR_EXT 0x80E0

#define MAX_MODEL_PATHS 120

/* Vector Structure*/  typedef struct { GLfloat n1, n2, n3;  } Vector;

/* Vertex Structure*/  typedef struct { GLfloat x,y,z; }       Vertex;

/* Vertex-Face Normal Structure*/ typedef struct { GLfloat n1,n2,n3; } Normal;

/*Texture Coordinates Structure*/ typedef struct { GLfloat u, v; }  TexCoords;

/* Face Structure*/
typedef struct
{
		long unsigned int v[3];
		long unsigned int n[3];
		long unsigned int t[3];
		Normal fc_normal;
		GLfloat area;
}Face;

/* Model Bounding Box Structure*/
typedef struct { Vertex min; Vertex max; } bbox;

/* Material Structure*/
typedef struct
{
        GLfloat shine;
		GLfloat ambient[4];
		GLfloat diffuse[4];
		GLfloat specular[4];
		char name[MAX_MODEL_PATHS];
		char texture[MAX_MODEL_PATHS];
		GLuint ldText;
		GLboolean hasTex; //has texture file
} Material;

/* Group Structure*/
typedef struct
{
	    long unsigned int *faceList;
		long unsigned int numFaces;
		GLuint material;
		GLboolean hasNormals;
		GLboolean hasTex; //has texture coords
		long unsigned int malloced;
		char name[MAX_MODEL_PATHS];
} Group;


typedef float MATRIX[16];

struct OBJ_Model
{
        bbox boundBox;

        //the list of vertices
		Vertex * vertexList;
		//the list of normals
		Normal * normalList;
		//the list of texture coordinates
		Face * faceList;
		//the number of faces
		TexCoords * texList;
		////the list of materials
		Material * matList;
		//the list of groups of our Mesh
		Group * groups;
		//the list of faces
		long unsigned int numFaces;

		GLfloat scale;
		//the center of our model
		GLfloat center[3];
		//the number of vertices
		long unsigned int numVertices;
		//the number of normals
		long unsigned int numNormals;
		//the number of texture coordinates
		long unsigned int numTexs;
		//the number of grups
		long unsigned int numGroups;
		//the number of materials
		long unsigned int numMaterials;


		//the name of the mtl file for the model
		char matLib[MAX_MODEL_PATHS];
		//the obj's filename
		char filename[MAX_MODEL_PATHS];
		//the display list id
		GLuint dispList;
};
GLuint getObjOGLList(struct OBJ_Model * obj);

struct OBJ_Model * loadObj(char * filename);
int unloadObj(struct OBJ_Model * obj);

void  drawOBJMesh(struct OBJ_Model * obj);
int findIntersection(struct OBJ_Model * obj,Vertex v1, Vertex v2, Vector* new_normal, Vector* intersection_point);

#endif // MODEL_LOADER_OBJ_H_INCLUDED
