//
// Created by tsiggi on 23/12/2023.
//


#pragma once

double* linspace(double start, double end, int numPoints) {
    if (numPoints <= 1) {
        double* result = new double[1];
        result[0] = start;
        return result;
    }
    
    double* result = new double[numPoints];
    double step = (end - start) / (numPoints - 1);
    for (int i = 0; i < numPoints; ++i) {
        result[i] = start + i * step;
    }
    
    return result;
}


