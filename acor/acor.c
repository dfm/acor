/*
 * This is a very slightly re-factored version of code by Jonathan Goodman
 * For the original code, see:
 *  http://www.math.nyu.edu/faculty/goodman/software/acor/index.html
 * Ported for use in Python with permission from the original author.
 */

#include <stdlib.h>
#include <math.h>

#include "acor.h"

#define WINMULT 5
#define MINFAC  5

int acor(double *mean, double *sigma, double *tau, double X[], int L, int maxlag)
{
    int i, s;
    double D;
    double *C;
    int iMax = L - maxlag;

    /* Declare variables for the pairwise analysis */
    int Lh = L/2, j1 = 0, j2 = 1;
    double newMean;

    /* Fail if the chain is too short */
    if (L < MINFAC*maxlag)
        return 1;

    /* Allocate memory for the auto-covariance function */
    if (!(C = (double*)malloc((maxlag+1)*sizeof(double))))
        return -1;

    /* compute the mean of X and normalize the data */
    *mean = 0.;
    for (i = 0; i < L; i++)
        *mean += X[i];
    *mean = *mean/L;
    for (i = 0; i <  L; i++ )
        X[i] -= *mean;

    /* Compute the auto-covariance function */
    for (s = 0; s <= maxlag; s++)
        C[s] = 0.;
    for (i = 0; i < iMax; i++)
        for (s = 0; s <= maxlag; s++)
            C[s] += X[i]*X[i+s];
    for (s = 0; s <= maxlag; s++)
        C[s] = C[s]/iMax; /* Normalize */

    /* Calculate the "diffusion coefficient" as the sum of the autocovariances */
    D = C[0];
    for (s = 1; s <= maxlag; s++ )
        D += 2*C[s];

    /* Calculate the standard error, if D were the complete sum. */
    /* FIXME: why is D sometimes negative?? */
    if ( D < 0 ) {
        return 2;
    }
    *sigma = sqrt( D / L );
    /* A provisional estimate, since D is only part of the complete sum. */
    *tau   = D / C[0];

    if ( *tau*WINMULT < maxlag ) {
        free(C);
        return 0; /* Stop if the D sum includes the given multiple of tau. */
    }

    /* If the provisional tau is so large that we don't think tau is accurate,
     * apply the acor procedure to the pairwise sums of X */
    for (i = 0; i < Lh; i++) {
        X[i] = X[j1] + X[j2];
        j1  += 2;
        j2  += 2;
    }

    acor( &newMean, sigma, tau, X, Lh, maxlag);
    D      = .25*(*sigma) * (*sigma) * L;
    *tau   = D/C[0];
    *sigma = sqrt( D/L );

    free(C);
    return 0;
}

