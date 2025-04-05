#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<cuda_runtime.h>
#include<iostream>
#include<string>
#include<fstream>

#include"Rendering.cu"

float gen_rand(float l, float u) {
    float s = rand();
    return (float)l + (u-l)*((float)s / RAND_MAX);
}

void saveToPPM(float* frameBuffer,const std::string& filename, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P3\n" << width << " " << height << "\n255\n";
    for(int h = 0; h < height;h++) {
        for(int w = 0;w < width;w++) {
            int idx = 3*(h * width + w);
            file<<static_cast<int>(frameBuffer[idx])<<" ";
            file<<static_cast<int>(frameBuffer[idx+1])<<" ";
            file<<static_cast<int>(frameBuffer[idx+2])<<" ";
        }
    }
    
    file.close();
}

int main() {
    const int width = 400;
    const int height = 300;
    const int numTriangles = 2;
    float ratio = (float)width / height;
    float z_far = 10.0f;
    float z_near = 0.1f;

    srand(time(NULL));

    Triangle* hostTriangles = new Triangle[numTriangles];
    float* hostColorBuffer = new float[width * height * 3];
    float* hostDepthBuffer = new float[width * height];

    //triangle 1
    hostTriangles[0].v0 = {
        Vec3(1/ratio, 0/ratio, -1),
        Vec3(0/ratio, 0, 1), //b 
        Vec3(0, 0, 1)
    };
    hostTriangles[0].v1 = {
        Vec3(0/ratio, 1/ratio, -1),
        Vec3(0, 1, 0), //g
        Vec3(0, 0, 1) 
    };
    hostTriangles[0].v2 = {
        Vec3(-1/ratio, 0/ratio, -1),
        Vec3(1, 0, 0), //r 
        Vec3(0, 0, 1) 
    };
    //triangle 2
    hostTriangles[1].v0 = {
        Vec3(0/ratio, -1/ratio, 0),
        Vec3(gen_rand(0, 1), gen_rand(0, 1), gen_rand(0, 1)), 
        Vec3(0, 0, 1)
    };
    hostTriangles[1].v1 = {
        Vec3(0.5/ratio, 0/ratio, 0),
        Vec3(gen_rand(0, 1), gen_rand(0, 1), gen_rand(0, 1)), 
        Vec3(0, 0, 1)
    };
    hostTriangles[1].v2 = {
        Vec3(-0.5/ratio, 0/ratio, 0),
        Vec3(gen_rand(0, 1), gen_rand(0, 1), gen_rand(0, 1)), 
        Vec3(0, 0, 1)
    };

    //Setup camera
    Camera camera;
    camera.position = Vec3(0, 0, 2);
    camera.target = Vec3(0, 0, 0);
    camera.up = Vec3(0, 1, 0);
    camera.fov = 45.0f * 3.141592f / 180.0f;
    camera.aspect = (float)width / height;
    camera.near = z_near;
    camera.far = z_far;

    //Rendering
    renderScene(hostTriangles, numTriangles, camera, hostColorBuffer, hostDepthBuffer, width, height);

    std::cout<<"Rendering complete"<<std::endl;

    //Save
    saveToPPM(hostColorBuffer, "output.ppm", width, height);

    //delete
    delete[] hostTriangles;
    delete[] hostColorBuffer;
    delete[] hostDepthBuffer;

    return 0;
}