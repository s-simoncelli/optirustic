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

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedParameter"
// Sort points improving in the last objective
bool greater(point *p, point *q, int fr, int obj_count) {
#if opt == 1
    for (int i = obj_count - fr - 1; i >= 0; i--)
#else
    for (int i = obj_count - 1; i >= 0; i--)
#endif
        if BEATS (p->objectives[i], q->objectives[i])
            return true;
        else if BEATS (q->objectives[i], p->objectives[i])
            return false;
    return false;
}
void sort_vec(point numbers[], int vec_size, front_set *fs, int obj_count) {
    bool did_swap;
    point temp;
    do {
        did_swap = false;
        for (int i = 0; i <= vec_size - 2; i++) {
            if (greater(&numbers[i], &numbers[i + 1], fs->fr, obj_count)) {
                did_swap = true;
                temp = numbers[i];
                numbers[i] = numbers[i + 1];
                numbers[i + 1] = temp;
            }
        }
    } while (did_swap);
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedParameter"
// Returns -1 if p dominates q, 1 if q dominates p, 2 if p == q, 0 otherwise
int dominates_2_way(point p, point q, int fr, int obj_count) {
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
#pragma clang diagnostic pop

// creates the front ps[p+1 ..] in fs[fr], with each point bounded by ps[p] and dominated points removed
void make_dominated_bit(front_set *fs, front ps, int p, int obj_count) {
    if (fs == NULL) {
        exit(1);
    }
    // when opt = 0 each new frame is allocated as needed, because the worst-case needs #frames = #points
#if opt == 0
    if (fs.fr > fs.fr_max || fs.fr == 0) {
        fs.fr_max = fs.fr;
        fs.sets[fs.fr].points = malloc(sizeof(point) * ps.number_of_individuals);
        for (int j = 0; j < ps.number_of_individuals; j++) {
            fs.sets[fs.fr].points[j].objectives = malloc(sizeof(double) * obj_count);
        }
    }
#endif

    int z = ps.number_of_individuals - 1 - p;
    for (int i = 0; i < z; i++)
        for (int j = 0; j < obj_count; j++)
            fs->sets[fs->fr].points[i].objectives[j] = WORSE(ps.points[p].objectives[j], ps.points[p + 1 + i].objectives[j]);
    point t;
    fs->sets[fs->fr].number_of_individuals = 1;

    for (int i = 1; i < z; i++) {
        int j = 0;
        bool keep = true;
        while (j < fs->sets[fs->fr].number_of_individuals && keep)
            switch (dominates_2_way(fs->sets[fs->fr].points[i], fs->sets[fs->fr].points[j], fs->fr, obj_count)) {
                case -1:
                    t = fs->sets[fs->fr].points[j];
                    fs->sets[fs->fr].number_of_individuals--;
                    fs->sets[fs->fr].points[j] = fs->sets[fs->fr].points[fs->sets[fs->fr].number_of_individuals];
                    fs->sets[fs->fr].points[fs->sets[fs->fr].number_of_individuals] = t;
                    break;
                case 0:
                    j++;
                    break;
                    // case  2: printf("Identical points!\n");
                default:
                    keep = false;
            }

        if (keep) {
            t = fs->sets[fs->fr].points[fs->sets[fs->fr].number_of_individuals];
            fs->sets[fs->fr].points[fs->sets[fs->fr].number_of_individuals] = fs->sets[fs->fr].points[i];
            fs->sets[fs->fr].points[i] = t;
            fs->sets[fs->fr].number_of_individuals++;
        }
    }
    fs->fr++;
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

// Return the inclusive hyper-volume of `p` with respect to the reference point `ref`
double inclusive_hv(point p, point ref, int obj_count) {
    double volume = 1;
    for (int i = 0; i < obj_count; i++)
        volume *= fabs(p.objectives[i] - ref.objectives[i]);
    return volume;
}


// Return the exclusive hyper-volume of `ps[p]` relative to ps[p+1 ..]
double exclusive_hv(front_set *fs, front ps, int p, point ref, int obj_count) {
    if (fs == NULL) {
        exit(1);
    }
    double volume = inclusive_hv(ps.points[p], ref, obj_count);
    if (ps.number_of_individuals > p + 1) {
        make_dominated_bit(fs, ps, p, obj_count);
        volume -= hv(fs, fs->sets[fs->fr - 1], ref, obj_count);
        fs->fr--;
    }
    return volume;
}


// Return the hyper-volume of `ps`
double hv(front_set *fs, front ps, point ref, int obj_count) {
#if opt > 0
#if DEBUG
    printf("Sorting\n");
#endif
    sort_vec(ps.points, ps.number_of_individuals, fs, obj_count);
#endif

#if opt == 2
    if (obj_count == 2)
        return hv_2d(ps, ref);
#endif

    double volume = 0;
#if opt <= 1
    for (int i = 0; i < ps.number_of_individuals; i++) {
        volume += exclusive_hv(fs, ps, i, ref, obj_count);
    }
#else
    obj_count--;
    for (int i = ps.number_of_individuals - 1; i >= 0; i--)
        // we can ditch dominated points here,
        // but they will be ditched anyway in dominatedBit
        volume += fabs(ps.points[i].objectives[obj_count] - ref.objectives[obj_count]) * exclusive_hv(fs, ps, i, ref, obj_count);
    obj_count++;
#endif
    return volume;
}

double calculate_hypervolume(front f, point ref) {
    int obj_count = f.number_of_objectives;
    front_set fs;
    fs.fr = 0;
    fs.fr_max = -1;

#if DEBUG
    printf("Number of objectives = %d\n", f.number_of_objectives);
    printf("Number of individuals = %d\n", f.number_of_individuals);
#endif

    // allocate memory
#if opt == 0
    fs.sets = malloc(sizeof(front) * f.number_of_individuals);
#else
    // slicing (opt > 1) saves a level of recursion
    int maxd = f.number_of_objectives - (opt / 2 + 1);
    fs.sets = malloc(sizeof(front) * maxd);

    // 3D base (opt = 3) needs space for the sentinels
    int maxp = f.number_of_individuals + 2 * (opt / 3);
    for (int i = 0; i < maxd; i++) {
        fs.sets[i].points = malloc(sizeof(point) * maxp);
        for (int j = 0; j < maxp; j++) {
            // slicing(opt > 1) saves one extra objective at each level
            fs.sets[i].points[j].objectives = malloc(sizeof(double) * (f.number_of_objectives - (i + 1) * (opt / 2)));
        }
    }
#endif

#if opt >= 3
    if (f.number_of_objectives == 2) {
        sort_vec(f.points, f.number_of_individuals, fs, f.number_of_objectives);
        return hv_2d(f, ref);
    } else
#endif
        return hv(&fs, f, ref, obj_count);
}