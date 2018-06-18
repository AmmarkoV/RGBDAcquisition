// Include standard headers
#include <stdio.h>
#include <stdlib.h>
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;


void printOutGLMMatrix(const char * label , glm::mat4 matrix)
{
  fprintf(stderr,"Martix `%s` : \n",label);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix[0][0],matrix[0][1],matrix[0][2],matrix[0][3]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix[1][0],matrix[1][1],matrix[1][2],matrix[1][3]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix[2][0],matrix[2][1],matrix[2][2],matrix[2][3]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f\n\n",matrix[3][0],matrix[3][1],matrix[3][2],matrix[3][3]);

}

int main( void )
{
	// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
	// Camera matrix
	glm::mat4 View       = glm::lookAt(
								glm::vec3(4,3,-3), // Camera is at (4,3,-3), in World Space
								glm::vec3(0,0,0), // and looks at the origin
								glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
						   );
	// Model matrix : an identity matrix (model will be at the origin)
	glm::mat4 Model      = glm::mat4(1.0f);
	// Our ModelViewProjection : multiplication of our 3 matrices
	glm::mat4 MVP        = Projection * View * Model; // Remember, matrix multiplication is the other way around


    printOutGLMMatrix("Projection",Projection);
    printOutGLMMatrix("View",View);
    printOutGLMMatrix("Model",Model);
    printOutGLMMatrix("MVP",MVP);

	return 0;
}

