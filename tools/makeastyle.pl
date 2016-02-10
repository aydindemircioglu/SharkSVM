#!/usr/bin/perl

if ( $#ARGV == -1 ) {
    die ("please specify which files to astyle.");
}

@flist = @ARGV;

$count = 0;
foreach $filename (@flist) {
	print "formatting file $filename:\n";
	print `astyle --options=none --style=java -d -f -CSNn -c -s4  -p -U -o -O $filename`;
	`echo mv $filename.orig /tmp`;
}
