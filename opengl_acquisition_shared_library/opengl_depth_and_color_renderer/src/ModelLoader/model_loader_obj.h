/** @file model_loader_obj.h
 *  @brief  A module that loads models from OBJ files
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef MODEL_LOADER_OBJ_H_INCLUDED
#define MODEL_LOADER_OBJ_H_INCLUDED

#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

#include "model_loader.h"

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


/**  @brief The structure that defines Vectors*/
typedef struct { GLfloat n1, n2, n3;  } Vector;

/**  @brief The structure that defines Vertexes*/
typedef struct { GLfloat x,y,z; }       Vertex;


/**  @brief The structure that defines Vertexes*/
typedef struct { GLfloat r,g,b; }       RGBColors;

/**  @brief The structure that defines Vertex-Face Normal  */
typedef struct { GLfloat n1,n2,n3; } Normal;


/**  @brief The structure that defines texture coordinates */
typedef struct { GLfloat u, v; }  TexCoords;


/**  @brief The structure that defines a Face ( its vertices normal , area etc ) */
typedef struct
{
		long unsigned int v[3];
		long unsigned int n[3];
		long unsigned int t[3];
		Normal fc_normal;
		GLfloat area;
}Face;


/**  @brief The structure that defines a Bounding Box */
typedef struct { Vertex min; Vertex max; } bbox;

/**  @brief The structure that defines a Material */
typedef struct
{
        GLfloat shine[1];
		GLfloat ambient[4];
		GLfloat diffuse[4];
		GLfloat specular[4];
		char name[MAX_MODEL_PATHS];
		char texture[MAX_MODEL_PATHS];
		GLuint ldText;
		GLboolean hasTex; //has texture file
} Material;

/**  @brief The structure that defines a Group */
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


/**  @brief The structure that defines a .OBJ Object */
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
		//the number of colors
		RGBColors * colorList;
		////the list of materials
		Material * matList;
		//the list of groups of our Mesh
		Group * groups;
		//the list of faces
		long unsigned int numFaces;

		GLfloat scale;
		//the center of our model
		GLfloat center[3];

        GLfloat minX,minY,minZ,maxX,maxY,maxZ;

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
		//the number of colors
		long unsigned int numColors;


		//the name of the mtl file for the model
		char matLib[MAX_MODEL_PATHS];
		//the obj's filename
		char filename[MAX_MODEL_PATHS];
		//the obj's directory
		char directory[MAX_MODEL_PATHS];
		//the display list id
		GLuint dispList;

		int customColor;
};




/**
* @brief Render a compiled OpenGL list using glCallList()
* @ingroup OBJModelLoader
* @param The loaded object
* @retval A pointer to a compiled opengl renderer
*/
GLuint getObjOGLList(struct OBJ_Model * obj);


/**
* @brief Load an Object (.OBJ) file
* @ingroup OBJModelLoader
* @param String with the directory of the model
* @param String with the filename of the model ( after the directory )
* @param Compile the object to an OpenGL DisplayList
* @retval 0=Failure , A pointer to an object model
*/
struct OBJ_Model * loadObj(char * directory,char * filename,int compileDisplayList);


/**
* @brief Unload a loaded Object
* @ingroup OBJModelLoader
* @param The loaded object
* @retval 0=Failure , 1=Success
*/
int unloadObj(struct OBJ_Model * obj);


/**
* @brief Draw a loaded Object
* @ingroup OBJModelLoader
* @param The loaded object
*/
void  drawOBJMesh(struct OBJ_Model * obj);


/**
* @brief Find intersection of vertexes with object model ( and return it )
* @ingroup OBJModelLoader
* @param The loaded object
* @param Vertex 1
* @param Vertex 2
* @param Output Normal
* @param Output Intersection point
* @retval A pointer to a compiled opengl renderer
*/
int findIntersection(struct OBJ_Model * obj,Vertex v1, Vertex v2, Vector* new_normal, Vector* intersection_point);

#endif // MODEL_LOADER_OBJ_H_INCLUDED
