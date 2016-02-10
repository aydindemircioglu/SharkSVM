#!/usr/bin/perl

use File::Temp qw/ tempfile tempdir /;
use File::Basename;


# this will try to remove as much SharkSVM from a class as possible


if ( $#ARGV == -1 ) {
    die ("please specify the files");
}
else {
    print "Specified files..\n";
    @flist = @ARGV;
    print @flist;
    print "\n";
}


foreach $filename (@flist) {
    ($fh, $tempfile) = tempfile();
    ($fh, $tempfile2) = tempfile();

    # correct header
    stop("correct header");
    
    
    # replace all SHARKSVMEXCEPTION by SHARKEXCEPTION
    `sed -e 's/SHARKSVMEXCEPTION/SHARKEXCEPTION/' $filename > $tempfile`;
    
    # remove all BOOST LOG lines
    `sed '/.*BOOST_LOG_/d' $tempfile > $tempfile2`;
    `cp $tempfile2 $tempfile`;

    # replace all include lines
    `sed '/\\#include.*SharkSVM.h.*/d' $tempfile > $tempfile2`;
    `cp $tempfile2 $tempfile`;
    
    # replace all include lines
    `sed '/\\#include.*GlobalParameters.h.*/d' $tempfile > $tempfile2`;
    `cp $tempfile2 $tempfile`;

        # replace all include lines
    `sed '/\\#include.*LibSVMDataModel.h.*/d' $tempfile > $tempfile2`;
    `cp $tempfile2 $tempfile`;

    # replace all include lines
    $p = `sed -e 's/\\#include.*\\(Budgeted.*h\\).*/#include <shark\\/Algorithms\\/Trainers\\/\\1>/' $tempfile > $tempfile2`;
    `cp $tempfile2 $tempfile`;

    # final replace
    $basename = fileparse($filename);
    `cp $tempfile shark/$basename`;
    
    # do some astyle the shark way
    `~/Shark/makeastyle shark/`;
}
 

