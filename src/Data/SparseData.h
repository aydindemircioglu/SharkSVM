//===========================================================================
/*!
 *
 *
 * \brief   Support for importing and exporting data from and to sparse data (libSVM) formatted data files
 *
 *
 * \par
 * The most important application of the methods provided in this
 * file is the import of data from LIBSVM files to Shark Data containers.
 *
 *
 *
 *
 * \author      M. Tuma, T. Glasmachers, C. Igel, Aydin Demircioglu
 * \date        2010
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

#ifndef SHARKSVM_DATA_SPARSEDATA_H
#define SHARKSVM_DATA_SPARSEDATA_H

#include <fstream>
#include <limits>

#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/adapted/std_pair.hpp>

#include <shark/Core/INameable.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Data/Dataset.h>

#include "SharkSVM.h"
#include "LabelOrder.h"


namespace shark {


    /// \brief A class that handles import and export of Sparse Data (also called
    /// LibSVM format sometimes).
    ///
    template <class InputType>
    class SparseDataModel : public INameable, ISerializable {
        private:

        public:

            typedef std::pair< unsigned int, size_t > LabelSortPair;
            typedef typename shark::LabeledData<InputType, unsigned int>::element_reference ElemRef;
            typedef std::pair<int, std::vector<std::pair<std::size_t, double> > > LibSVMPoint;


            /// \brief  for sorting in decreasing order
            static inline bool cmpLabelSortPair (const  LabelSortPair& left, const LabelSortPair& right) {
                return left.first > right.first;
            }


            /// \brief constructor
            SparseDataModel()  {};


            /// \brief destructor
            virtual ~SparseDataModel() {};



            /// \brief Export data to sparse data (libSVM) format.
            ///
            /// \param  dataset     Container storing the  data
            /// \param  fn          Output file
            /// \param  dense       Flag for using dense output format
            /// \param  oneMinusOne Flag for applying the transformation y<-2y-1 to binary labels
            /// \param  sortLabels  Flag for sorting data points according to labels
            /// \param  append      Flag for appending to the output file instead of overwriting it
            ///
            void exportData (LabeledData<InputType, unsigned int> const &dataset,
                             const std::string &fn,
                             bool dense = false,
                             bool oneMinusOne = true,
                             bool sortLabels = false,
                             bool append = false) {
                std::size_t elements = dataset.numberOfElements();
                std::ofstream ofs;

                // shall we append only or overwrite?
                if (append == true) {
                    ofs.open (fn.c_str(), std::fstream::out | std::fstream::app);
                } else {
                    ofs.open (fn.c_str());
                }

                if (!ofs) {
                    throw (SHARKSVMEXCEPTION ("File can not be opened for writing"));
                }

                size_t dim = inputDimension (dataset);

                if (numberOfClasses (dataset) != 2) oneMinusOne = false;

                std::vector<LabelSortPair> L;

                if (sortLabels) {
                    for (std::size_t i = 0; i < elements; i++)
                        L.push_back (LabelSortPair (dataset.element (i).label, i));

                    std::sort (L.begin(), L.end(), cmpLabelSortPair);
                }

                for (std::size_t ii = 0; ii < elements; ii++) {
                    // apply mapping to sorted indices
                    std::size_t i = 0;

                    if (sortLabels) i = L[ii].second;
                    else i = ii;

                    // apply transformation to label and write it to file
                    if (oneMinusOne) ofs << 2 * int (dataset.element (i).label) - 1 << " ";
                    //libsvm file format documentation is scarce, but by convention the first class seems to be 1..
                    else ofs << dataset.element (i).label + 1 << " ";

                    // write input data to file
                    for (std::size_t j = 0; j < dim; j++) {
                        if (dense)
                            ofs << " " << j + 1 << ":" << dataset.element (i).input (j);
                        else if (dataset.element (i).input (j) != 0)
                            ofs << " " << j + 1 << ":" << dataset.element (i).input (j);
                    }

                    ofs << std::endl;
                }
            }



            /// \brief Import data from sparse data (libSVM) format.
            /// this differs from the shark implementation in one way:
            /// labels will be kept as they are
            ///
            /// \param  dataset       container storing the loaded data
            /// \param  stream        stream to be read from
            /// \param  labelOrder      the label order extracted from the data
            /// \param  normalizeLabels     if true, the labels will be normalized,
            ///                                                    if false, the labels will not be modified.
            /// \param  dimensions  highest feature index, or 0 for auto-detection
            /// \param  batchSize     size of batch
            ///
            LabeledData<RealVector, unsigned int> importData (
                std::istream& stream,
                LabelOrder &labelOrder,
                bool normallizeLabels = true,
                unsigned int dimensions = 0,
                std::size_t batchSize = LabeledData<InputType, unsigned int>::DefaultBatchSize) {
                //read contents of stream
                std::vector<LibSVMPoint> contents = importSparseDataReader (stream);
                std::size_t numPoints = contents.size();

                //find data dimension by getting the maximum index
                std::size_t maxIndex = 0;
                for (std::size_t i = 0; i != numPoints; ++i) {
                    std::vector<std::pair<std::size_t, double> > const& inputs = contents[i].second;

                    if (!inputs.empty())
                        maxIndex = std::max (maxIndex, inputs.back().first);
                }


                // did we specify a dimension?
                if (dimensions == 0) {
                    // no, so we take what the max index/dimension we saw
                    dimensions = maxIndex;
                } else {
                    // check if the given dimension matches
                    if (maxIndex > dimensions)
                        throw SHARKSVMEXCEPTION ("Number of dimensions supplied is smaller than actual index data");
                }


                //check labels for conformity
                bool binaryLabels = false;
                int minPositiveLabel = std::numeric_limits<int>::max();
                {
                    int maxPositiveLabel = -1;

                    for (std::size_t i = 0; i != numPoints; ++i) {
                        int label = contents[i].first;

                        if (label < -1)
                            throw SHARKSVMEXCEPTION ("Negative labels are only allowed for classes -1/1");
                        else if (label == -1)
                            binaryLabels = true;
                        else if (label < minPositiveLabel)
                            minPositiveLabel = label;
                        else if (label > maxPositiveLabel)
                            maxPositiveLabel = label;
                    }

                    if (binaryLabels && (minPositiveLabel == 0 ||  maxPositiveLabel > 1))
                        throw SHARKSVMEXCEPTION ("Negative labels are only allowed for classes -1/1");
                }

                // check for feature index zero (non-standard, but it happens)
                bool haszero = false;

                for (std::size_t i = 0; i < numPoints; i++) {
                    std::vector<std::pair<std::size_t, double> > const& input = contents[i].second;

                    if (input.empty()) continue;

                    if (input[0].first == 0) {
                        haszero = true;
                        break;
                    }
                }


                // copy contents into a new dataset
                // and one normalize labels on the fly

                // array that tracks what we already encountered
                std::vector<int> tmpLabelOrder;

                // create dataset with the right structure
                typename shark::LabeledData<InputType, unsigned int>::element_type blueprint (InputType (maxIndex + (haszero ? 1 : 0)), 0);
                shark::LabeledData<InputType, unsigned int> data (numPoints, blueprint, batchSize);

                size_t delta = (haszero ? 0 : 1);
                std::size_t i = 0;
                BOOST_FOREACH (ElemRef element, data.elements()) {
                    element.input.clear();
                    int tmpLabel = contents[i].first;

                    // do we know that label already?
                    if (std::find (tmpLabelOrder.begin(), tmpLabelOrder.end(), tmpLabel) == tmpLabelOrder.end()) {
                        // no, so we add it to the tmpLabelOrder
                        tmpLabelOrder.push_back (tmpLabel);
                    }

                    // now we can assume that the label is already in our ordering.
                    // so we reverse-map the label
                    std::vector<int>::iterator pos = std::find (tmpLabelOrder.begin(), tmpLabelOrder.end(), tmpLabel);
                    unsigned int mappedLabel = pos - tmpLabelOrder.begin() ;

                    // if we want to normalize the labels, we overwrite the default value
                    element.label = tmpLabel;
                    if (normallizeLabels == true)
                        element.label = mappedLabel;
                    
                    // copy over the components of the current vector
                    std::vector<std::pair<std::size_t, double> > const& inputs = contents[i].second;
                    for (std::size_t j = 0; j != inputs.size(); ++j)
                        element.input (inputs[j].first - delta) = inputs[j].second; 

                    ++i;
                }

                // TODO: sanity check for one-class?

                if ((tmpLabelOrder.size() != 2) && (binaryLabels == true))
                    throw SHARKSVMEXCEPTION ("Negative label indicated binary data, but found more than two labels.");
                

                // finally add our new found labels to the ordering we were given
                labelOrder.setLabelOrder (tmpLabelOrder);

                return (data);
            }



            /// \brief Import data from sparse data (libSVM) format.
            ///
            /// \param  dataset       container storing the loaded data
            /// \param  fn      filename
            /// \param  normalizeLabels     if true, the labels will be normalized,
            ///                                                    if false, the labels will not be modified.
            /// \param  dimensions  highest feature index, or 0 for auto-detection
            /// \param  batchSize     size of batch
            ///
            LabeledData<RealVector, unsigned int> importData (
                std::string fn,
                bool normallizeLabels = true,
                unsigned int dimensions = 0,
                std::size_t batchSize = LabeledData<InputType, unsigned int>::DefaultBatchSize) {
                std::ifstream ifs (fn.c_str());

                if (! ifs.good())
                    throw SHARKSVMEXCEPTION ("Failed to open file for input");

                // create a dummy labelorder and forget it again
                LabelOrder labelOrder;
                return importData (ifs, labelOrder, normallizeLabels, dimensions, batchSize);
            }



            /// \brief Import data from sparse data (libSVM) format.
            ///
            /// \param  dataset       container storing the loaded data
            /// \param  fn      filename
            /// \param  normalizeLabels     if true, the labels will be normalized,
            ///                                                    if false, the labels will not be modified.
            /// \param  dimensions  highest feature index, or 0 for auto-detection
            /// \param  batchSize     size of batch
            ///
            LabeledData<RealVector, unsigned int> importData (
                std::string fn,
                LabelOrder &labelOrder,
                bool normallizeLabels = true,
                unsigned int dimensions = 0,
                std::size_t batchSize = LabeledData<InputType, unsigned int>::DefaultBatchSize) {
                std::ifstream ifs (fn.c_str());

                if (! ifs.good())
                    throw SHARKSVMEXCEPTION ("Failed to open file for input");

                return importData (ifs, labelOrder, normallizeLabels, dimensions, batchSize);
            }


        private:

            inline std::vector<LibSVMPoint>
            importSparseDataReader (
                std::istream& stream) {
                std::vector<LibSVMPoint>  fileContents;

                while (stream) {
                    std::string line;
                    std::getline (stream, line);

                    if (line.empty()) continue;

                    using namespace boost::spirit::qi;
                    std::string::const_iterator first = line.begin();
                    std::string::const_iterator last = line.end();

                    LibSVMPoint newPoint;
                    bool r = phrase_parse (
                                 first, last,
                                 int_   >> - (lit ('.') >> +lit ('0')) //we also want to be able to parse 1.00000 as label 1
                                 >> * (uint_ >> ':' >> double_),
                                 space , newPoint
                             );

                    if (!r || first != last) {
                        std::cout << std::string (first, last) << std::endl;
                        throw SHARKSVMEXCEPTION ("Problems parsing the sparse data file.");
                    }

                    fileContents.push_back (newPoint);
                }

                return fileContents;
            }


    };

} // namespace detail


#endif
