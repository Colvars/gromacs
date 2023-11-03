/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2023- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
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
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/* This file was inspired by ch5md by Pierre de Buyl (BSD license). */

#ifndef GMX_FILEIO_H5MD_IO_H
#define GMX_FILEIO_H5MD_IO_H

#include <string>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"
#include "h5md_datablock.h"

struct gmx_mtop_t;
typedef int64_t hid_t;
typedef unsigned long long hsize_t;

/*! \brief The container of the H5MD data. The class is designed to read/write data according to de Buyl et al., 2014
 * (https://www.sciencedirect.com/science/article/pii/S0010465514000447) and https://www.nongnu.org/h5md/h5md.html
 * The class contains a number of standard data blocks that are commonly used by GROMACS. */
class GmxH5mdIo
{
private:
    hid_t   file_;              //!< The HDF5 identifier of the file. This is the H5MD root.
    GmxH5mdDataBlock position_; //!< The data block with lossless positions.
    GmxH5mdDataBlock positionLossy_; //!< The data block with compressed (lossy compression) positions.
    GmxH5mdDataBlock velocity_; //!< The data block with lossless velocities.
    GmxH5mdDataBlock force_; //!< The data block with lossless forces.
    GmxH5mdDataBlock box_; //!< The data block with the box shape written together with lossless positions.
    GmxH5mdDataBlock boxLossy_; //!< The data block with the box shape written together with lossy (compressed) positions.
    GmxH5mdDataBlock atomName_; //!< A data block with the names of all atoms in the system.
    GmxH5mdDataBlock atomType_; //!< A data block with the atom type of all atoms in the system.
    GmxH5mdDataBlock charge_; //!< A data block with the partial charges of all atoms in the system.
    GmxH5mdDataBlock mass_; //!< A data block with the atom masses of all atoms in the system.

    /*! \brief Sets the author (user) and creator (application name) properties in the h5md group (h5mdGroup_). */
    void setAuthorAndCreator();

public:
    /*! \brief Construct a GmxH5mdIo object and open a GmxHdf5 file.
     *
     * \param[in] fileName    Name of the file to open. The same as the file path.
     * \param[in] mode        The mode to open the file, described by a lower-case letter
     *                        'w' means writing (and reading), i.e. backup an existing file and replace it.
     *                        'a' means appending (and reading), i.e., that existing files will be not be overwritten, but extended.
     *                        'r' means only reading.
     */
    GmxH5mdIo(const char* fileName = "", const char mode = '\0');

    ~GmxH5mdIo();

    /*! \brief Open an H5MD file.
     *
     * \param[in] fileName    Name of the file to open. The same as the file path.
     * \param[in] mode        The mode to open the file, described by a case-insensitive string of
     *                        letters, up to three characters long. Reading is always assumed.
     *                        'w' means writing, i.e. backup an existing file and replace it,
     *                        'a' means truncate, i.e., that existing files will be overwritten
     *                        'r' means only read.
     */
    void openFile(const char* fileName, const char mode);

    /*! \brief Close the H5MD file. */
    void closeFile();

    /*! \brief Write all unwritten data to the file. */
    void flush();

    /*! \brief Write molecule system related data to the file.
     *
     * This is currently not updated during the trajectory. The data that is written are atom masses, atom charges and atom names.
     *
     * \param[in] topology The molecular topology describing the system.
     */
    void setupMolecularSystem(const gmx_mtop_t& topology);

    /*! \brief Set up data blocks related to particle data.
     *
     * This needs to be done before writing the particle data to the trajectory.
     *
     * \param[in] writeCoordinatesSteps The lossless coordinate output interval.
     * \param[in] writeCoordinatesCompressedSteps The lossy compressed coordinate output interval.
     * \param[in] writeForcesSteps The lossless force output interval.
     * \param[in] writeVelocitiesSteps The lossless velocity output interval.
     * \param[in] numParticles The number of particles/atoms in the system.
     * \param[in] numParticlesCompressed The number of particles/atoms used for writing compressed coordinate data.
     * \param[in] compressionError The required precision of the lossy compression.
     */
    void setUpParticlesDataBlocks(int writeCoordinatesSteps, int writeCoordinatesCompressedSteps, int writeForcesSteps, int writeVelocitiesSteps, int numParticles, int numParticlesCompressed, double compressionError);

    /*! \brief Write a trajectory frame to the file. Only writes the data that is passed as input
     *
     * \param[in] step The simulation step.
     * \param[in] time The time stamp (in ps).
     * \param[in] lambda The lambda state. FIXME: Currently not written.
     * \param[in] box The box dimensions.
     * \param[in] x The particle coordinates for lossless output.
     * \param[in] v The particle velocities for lossless output.
     * \param[in] f The particle forces for lossless output.
     * \param[in] xLossy The particle coordinates for lossy (compressed) output.
     */
    void writeFrame(int64_t          step,
                    real             time,
                    real             lambda,
                    const rvec*      box,
                    const rvec*      x,
                    const rvec*      v,
                    const rvec*      f,
                    const rvec*      xLossy);
};

#endif // GMX_FILEIO_H5MD_IO_H