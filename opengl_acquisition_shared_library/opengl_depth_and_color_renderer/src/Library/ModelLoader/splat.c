// gcc splat.c -o splat
//This is a tool (that doesnt work yet..) to convert gaussian splatting files to regular ply
//https://github.com/graphdeco-inria/gaussian-splatting/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    float x, y, z;
    float nx, ny, nz;
    float f_dc_0, f_dc_1, f_dc_2;
    float f_rest[45];
    float opacity;
    float scale[3];
    float rot[4];
    unsigned char r, g, b;
} Vertex;



// Function to convert spherical harmonics to RGB color
void sphericalHarmonicToRGB(float sh_r, float sh_g, float sh_b, unsigned char* r, unsigned char* g, unsigned char* b) {
    // Scale and clamp the SH values to the 0-255 range
    *r = (unsigned char)(sh_r * 255.0f);
    *g = (unsigned char)(sh_g * 255.0f);
    *b = (unsigned char)(sh_b * 255.0f);
}





int main() {
    FILE* ply_file = fopen("point_cloud.ply", "rb");

    if (!ply_file) {
        perror("Failed to open PLY file");
        return 1;
    }

    // Skip header information
    char line[256];
    while (fgets(line, sizeof(line), ply_file)) {
        if (strcmp(line, "end_header\n") == 0) {
            break;
        }
    }

    // Read vertices
    int num_vertices = 585563;
    Vertex* vertices = (Vertex*)malloc(num_vertices * sizeof(Vertex));

    if (fread(vertices, sizeof(Vertex), num_vertices, ply_file) != num_vertices) {
        perror("Failed to read vertices");
        fclose(ply_file);
        free(vertices);
        return 1;
    }

    fclose(ply_file);


    for (int i = 0; i < num_vertices; i++) {
        // Convert spherical harmonics to RGB color
        float sh_r = vertices[i].f_dc_0;
        float sh_g = vertices[i].f_dc_1;
        float sh_b = vertices[i].f_dc_2;

        sphericalHarmonicToRGB(sh_r, sh_g, sh_b, &vertices[i].r, &vertices[i].g, &vertices[i].b);
    }


    FILE* ply_file_output = fopen("output_model_with_colors.ply", "wb");
    if (!ply_file_output) {
        perror("Failed to open output PLY file");
        free(vertices);
        return 1;
    }

    // Write PLY header
    fprintf(ply_file_output, "ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n", num_vertices);

    // Write vertices with RGB colors
    for (int i = 0; i < num_vertices; i++) {
        fprintf(ply_file_output, "%f %f %f %d %d %d\n", vertices[i].x, vertices[i].y, vertices[i].z, vertices[i].r, vertices[i].g, vertices[i].b);
    }

    fclose(ply_file_output);


    // Now you have the vertices and their properties in the 'vertices' array.
    // You can use this data for rendering or other purposes.

    // Don't forget to free the allocated memory when you're done.
    free(vertices);

    return 0;
}

