#ifndef _WFG_H_
#define _WFG_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int greater(const void *v1, const void *v2);


///A struct defining the objective values for an individual
typedef struct {
    ///The list of objective values
    double *objectives;
} point;

///A struct defining the front with the objective valued.
typedef struct {
    ///The number of points or individuals in the front
    int number_of_individuals;
    ///The number of objectives
    int number_of_objectives;
    ///The points
    point *points;
} front;

double hv_2d(front, point);
double hv(front, point);

///Calculate the hyper-volume with the WFG algorithm
double calculate_hypervolume(front, point);

#endif
