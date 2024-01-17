#include <stdio.h>
#include <math.h>
//gcc ray.c -lm -o ray && ./ray 
#define WIDTH 800
#define HEIGHT 600

typedef struct {
    double x, y, z;
} Vec3;

typedef struct {
    Vec3 origin;
    Vec3 direction;
    double polarization_angle;
} Ray;

typedef struct {
    Vec3 position;
    double radius;
} Sphere;

typedef struct 
{
    unsigned char r, g, b;
} Color;

typedef struct {
    Vec3 position;  // Dynamic position of the light source
    Color color;
} Light;


Color image[WIDTH][HEIGHT];

Vec3 subtract(Vec3 a, Vec3 b) {
    Vec3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

int hit_sphere(Sphere sphere, Ray ray) {
    Vec3 oc = subtract(ray.origin, sphere.position);
    double a = dot(ray.direction, ray.direction);
    double b = 2.0 * dot(oc, ray.direction);
    double c = dot(oc, oc) - sphere.radius * sphere.radius;
    double discriminant = b * b - 4 * a * c;
    return (discriminant > 0);
}

Color ray_color(Ray ray, Light light) {
    Sphere sphere;
    sphere.position.x = 0;
    sphere.position.y = 0;
    sphere.position.z = -1;
    sphere.radius = 0.5;

    if (hit_sphere(sphere, ray)) {
        double polarization_angle = ray.polarization_angle;
        double cos_theta = fabs(dot(ray.direction, subtract(sphere.position, ray.origin)));
        double intensity = cos(polarization_angle - cos_theta);
        intensity = fmax(0, intensity);

        // Calculate light direction and distance
        Vec3 light_direction = subtract(light.position, sphere.position);
        double light_distance = sqrt(dot(light_direction, light_direction));
        light_direction.x /= light_distance;
        light_direction.y /= light_distance;
        light_direction.z /= light_distance;

        // Calculate Lambertian reflection (diffuse reflection)
        double lambertian = fmax(0.0, dot(light_direction, ray.direction));

        // Combine Lambertian reflection with intensity
        intensity *= lambertian;

        Color hit_color = {255, 0, 0};  // Red color if the ray hits the sphere
        hit_color.r *= intensity * light.color.r;
        hit_color.g *= intensity * light.color.g;
        hit_color.b *= intensity * light.color.b;

        return hit_color;
    }

    // Background color (black)
    return (Color){0, 0, 0};
}

void render(Light light) {
    double aspect_ratio = (double)WIDTH / HEIGHT;
    double viewport_height = 2.0;
    double viewport_width = aspect_ratio * viewport_height;
    double focal_length = 1.0;

    Vec3 origin = {1, 0.3, 0};
    Vec3 horizontal = {viewport_width, 0, 0};
    Vec3 vertical = {0, viewport_height, 0};
    Vec3 lower_left_corner = subtract(origin, horizontal);
    lower_left_corner = subtract(lower_left_corner, vertical);
    lower_left_corner.z = lower_left_corner.z - focal_length;

    for (int j = HEIGHT - 1; j >= 0; --j) {
        for (int i = 0; i < WIDTH; ++i) {
            double u = (double)i / (WIDTH - 1);
            double v = (double)j / (HEIGHT - 1);

            Vec3 direction = subtract(lower_left_corner, origin);
            direction.x = direction.x + u * horizontal.x + v * vertical.x;
            direction.y = direction.y + u * horizontal.y + v * vertical.y;
            direction.z = direction.z + u * horizontal.z + v * vertical.z;

            double polarization_angle = M_PI / 4.0;  // Default polarization angle
            Ray ray = {origin, direction, polarization_angle};
            Color pixel_color = ray_color(ray, light);

            image[i][j] = pixel_color;
        }
    }
}

void save_image() {
    FILE *file = fopen("output.ppm", "w");
    fprintf(file, "P3\n%d %d\n255\n", WIDTH, HEIGHT);

    for (int j = HEIGHT - 1; j >= 0; --j) {
        for (int i = 0; i < WIDTH; ++i) {
            fprintf(file, "%d %d %d ", image[i][j].r, image[i][j].g, image[i][j].b);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main() {
    Light light = {{0.5, 0.5, -2}, {1, 1, 1}};  // Set dynamic light position and color
    render(light);
    save_image();
    return 0;
}

