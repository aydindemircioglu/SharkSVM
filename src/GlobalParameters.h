//===========================================================================
/*!
 *
 *
 * \brief       A class keeping all global parameters.
 *
 * \par  
 * This is basically a singleton, as it will only be called via boost::serialization::singleton.
 *
 * \author      Aydin Demircioglu
 * \date        2014
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 *
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================


#ifndef SHARK_GLOBALPARAMETERS_H
#define SHARK_GLOBALPARAMETERS_H

#include <sys/time.h> 

// for now stupid defines.. FIXME

#define SAFETYWALLTIME  5

// check for walltime every ...th iteration
#define CHECKINTERVAL 10

// how to convert given times into seconds, 60=minutes, 1=seconds
#define TIMEFACTOR 60




class GlobalParameters {
    public:
        // control parameter
        std::string modelPath;
        int32_t wallTime;
        int32_t saveTime;


        double now() const {
            struct timeval t;
            gettimeofday (&t, NULL);
            return (t.tv_sec + 1e-6 * t.tv_usec);
        }


};


#endif
