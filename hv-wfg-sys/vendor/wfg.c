/*

 This program is free software (software libre); you can redistribute
 it and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, you can obtain a copy of the GNU
 General Public License at:
                 http://www.gnu.org/copyleft/gpl.html
 or by writing to:
           Free Software Foundation, Inc., 59 Temple Place,
                 Suite 330, Boston, MA 02111-1307 USA

 ----------------------------------------------------------------------

*/

// opt:  0 = basic, 1 = sorting, 2 = slicing to 2D, 3 = slicing to 3D

#include "wfg.h"
#include <math.h>
#include <stdbool.h>

#if MAXIMISING
#define BEATS(x, y) (x > y)
#define BEATSEQ(x, y) (x >= y)
#else
#define BEATS(x, y) (x < y)
#define BEATSEQ(x, y) (x <= y)
#endif

#define WORSE(x, y) (BEATS(y, x) ? (x) : (y))
#define BETTER(x, y) (BEATS(y, x) ? (y) : (x))

int obj_count; // the number of objectives

/// Initialise the structure to track fronts
front *fs;
/// Current depth
int fr = 0;
/// Max depth malloced so far (for opt = 0)
int frmax = -1;

// This sorts points improving in the last objective
int greater(const void *v1, const void *v2) {
    point p = *(point *) v1;
    point q = *(point *) v2;
#if opt == 1
    for (int i = obj_count - fr - 1; i >= 0; i--)
#else
    for (int i = obj_count - 1; i >= 0; i--)
#endif
        if BEATS (p.objectives[i], q.objectives[i])
            return 1;
        else if BEATS (q.objectives[i], p.objectives[i])
            return -1;
    return 0;
}

// Returns -1 if p dominates q, 1 if q dominates p, 2 if p == q, 0 otherwise
int dominates2way(point p, point q) {
    // domination could be checked in either order
#if opt == 1
    for (int i = obj_count - fr - 1; i >= 0; i--)
#else
    for (int i = obj_count - 1; i >= 0; i--)
#endif
        if BEATS (p.objectives[i], q.objectives[i]) {
            for (int j = i - 1; j >= 0; j--)
                if BEATS (q.objectives[j], p.objectives[j])
                    return 0;
            return -1;
        } else if BEATS (q.objectives[i], p.objectives[i]) {
            for (int j = i - 1; j >= 0; j--)
                if BEATS (p.objectives[j], q.objectives[j])
                    return 0;
            return 1;
        }
    return 2;
}

// creates the front ps[p+1 ..] in fs[fr], with each point bounded by ps[p] and dominated points removed
void makeDominatedBit(front ps, int p) {
    // when opt = 0 each new frame is allocated as needed, because the worst-case needs #frames = #points
#if opt == 0
    if (fr > frmax || fr == 0) {
        frmax = fr;
        fs[fr].points = malloc(sizeof(point) * ps.number_of_individuals);
        for (int j = 0; j < ps.number_of_individuals; j++) {
            fs[fr].points[j].objectives = malloc(sizeof(double) * obj_count);
        }
    }
#endif

    int z = ps.number_of_individuals - 1 - p;
    for (int i = 0; i < z; i++)
        for (int j = 0; j < obj_count; j++)
            fs[fr].points[i].objectives[j] = WORSE(ps.points[p].objectives[j], ps.points[p + 1 + i].objectives[j]);
    point t;
    fs[fr].number_of_individuals = 1;

    for (int i = 1; i < z; i++) {
        int j = 0;
        bool keep = true;
        while (j < fs[fr].number_of_individuals && keep)
            switch (dominates2way(fs[fr].points[i], fs[fr].points[j])) {
                case -1:
                    t = fs[fr].points[j];
                    fs[fr].number_of_individuals--;
                    fs[fr].points[j] = fs[fr].points[fs[fr].number_of_individuals];
                    fs[fr].points[fs[fr].number_of_individuals] = t;
                    break;
                case 0:
                    j++;
                    break;
                    // case  2: printf("Identical points!\n");
                default:
                    keep = false;
            }

        if (keep) {
            t = fs[fr].points[fs[fr].number_of_individuals];
            fs[fr].points[fs[fr].number_of_individuals] = fs[fr].points[i];
            fs[fr].points[i] = t;
            fs[fr].number_of_individuals++;
        }
    }
    fr++;
}

// Returns the hyper-volume of ps[0 ..] in 2D
double hv_2d(front ps, point ref)
// assumes that ps is sorted improving
{
    double volume = fabs(
            (ps.points[0].objectives[0] - ref.objectives[0]) * (ps.points[0].objectives[1] - ref.objectives[1]));
    for (int i = 1; i < ps.number_of_individuals; i++)
        volume += fabs((ps.points[i].objectives[0] - ref.objectives[0]) *
                       (ps.points[i].objectives[1] -
                        ps.points[i - 1].objectives[1]));
    return volume;
}

// Returns the inclusive hypervolume of p
double inclusive_hv(point p, point ref) {
    double volume = 1;
    for (int i = 0; i < obj_count; i++)
        volume *= fabs(p.objectives[i] - ref.objectives[i]);
    return volume;
}


// Returns the exclusive hypervolume of ps[p] relative to ps[p+1 ..]
double exclusive_hv(front ps, int p, point ref) {
    double volume = inclusive_hv(ps.points[p], ref);
    if (ps.number_of_individuals > p + 1) {
        makeDominatedBit(ps, p);
        volume -= hv(fs[fr - 1], ref);
        fr--;
    }
    return volume;
}


// returns the hypervolume of ps[0 ..]
double hv(front ps, point ref) {
#if opt > 0
#if DEBUG
    printf("Sorting\n");
#endif
    qsort(ps.points, ps.number_of_individuals, sizeof(point), greater);
#endif

#if opt == 2
    if (obj_count == 2)
        return hv_2d(ps, ref);
#endif

    double volume = 0;
#if opt <= 1
    for (int i = 0; i < ps.number_of_individuals; i++) {
        double exclhv_ind = exclusive_hv(ps, i, ref);
        volume += exclhv_ind;
    }
#else
    obj_count--;
    for (int i = ps.number_of_individuals - 1; i >= 0; i--)
        // we can ditch dominated points here,
        // but they will be ditched anyway in dominatedBit
        volume += fabs(ps.points[i].objectives[obj_count] - ref.objectives[obj_count]) * exclusive_hv(ps, i, ref);
    obj_count++;
#endif
    return volume;
}

double calculate_hypervolume(front f, point ref) {
    // number of objectives
    obj_count = f.number_of_objectives;

#if DEBUG
    printf("Number of objectives = %d\n", obj_count);
    printf("Number of individuals = %d\n", f.number_of_individuals);
#endif

    // allocate memory
#if opt == 0
    fs = malloc(sizeof(front) * f.number_of_individuals);
#else
    // slicing (opt > 1) saves a level of recursion
    int maxd = obj_count - (opt / 2 + 1);
    fs = malloc(sizeof(front) * maxd);

    // 3D base (opt = 3) needs space for the sentinels
    int maxp = f.number_of_individuals + 2 * (opt / 3);
    for (int i = 0; i < maxd; i++) {
        fs[i].points = malloc(sizeof(point) * maxp);
        for (int j = 0; j < maxp; j++) {
            // slicing(opt > 1) saves one extra objective at each level
            fs[i].points[j].objectives = malloc(sizeof(double) * (obj_count - (i + 1) * (opt / 2)));
        }
    }
#endif

#if opt >= 3
    if (obj_count == 2) {
        qsort(f.points, f.number_of_individuals, sizeof(point), greater);
        return hv_2d(f, ref);
    } else
#endif
        return hv(f, ref);
}