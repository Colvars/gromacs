/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2019,2020,2021, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

/*! \internal \file
 * \brief
 * Implements the PairSearch class
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#include "gmxpre.h"

#include "pairsearch.h"

#include "gromacs/mdtypes/nblist.h"

#include "pairlist.h"


void SearchCycleCounting::printCycles(FILE* fp, gmx::ArrayRef<const PairsearchWork> work) const
{
    fprintf(fp, "\n");
    fprintf(fp,
            "ns %4d grid %4.1f search %4.1f",
            cc_[enbsCCgrid].count(),
            cc_[enbsCCgrid].averageMCycles(),
            cc_[enbsCCsearch].averageMCycles());

    if (work.size() > 1)
    {
        if (cc_[enbsCCcombine].count() > 0)
        {
            fprintf(fp, " comb %5.2f", cc_[enbsCCcombine].averageMCycles());
        }
        fprintf(fp, " s. th");
        for (const PairsearchWork& workEntry : work)
        {
            fprintf(fp, " %4.1f", workEntry.cycleCounter.averageMCycles());
        }
    }
    fprintf(fp, "\n");
}

#ifndef DOXYGEN

PairsearchWork::PairsearchWork() :
    cp0({ { 0 } }), ndistc(0), nbl_fep(std::make_unique<t_nblist>()), cp1({ { 0 } })
{
}

#endif // !DOXYGEN

PairsearchWork::~PairsearchWork() = default;

PairSearch::PairSearch(const PbcType             pbcType,
                       const bool                doTestParticleInsertion,
                       const ivec*               numDDCells,
                       const gmx_domdec_zones_t* ddZones,
                       const PairlistType        pairlistType,
                       const bool                haveFep,
                       const int                 maxNumThreads,
                       gmx::PinningPolicy        pinningPolicy) :
    gridSet_(pbcType, doTestParticleInsertion, numDDCells, ddZones, pairlistType, haveFep, maxNumThreads, pinningPolicy),
    work_(maxNumThreads)
{
    cycleCounting_.recordCycles_ = (getenv("GMX_NBNXN_CYCLE") != nullptr);
}
