$pdf_mode = 1;
$out_dir = 'build';
$aux_dir = 'build';
$do_cd = 1;
$pdflatex = 'pdflatex -synctex=1 %O %S';

use Fcntl qw(:flock);
use Cwd qw(abs_path);
use File::Basename qw(dirname);
use File::Path qw(make_path);
our $companion_build_lock;
unless (defined $companion_build_lock) {
    my $build_dir = dirname(abs_path(__FILE__)) . '/build';
    make_path($build_dir);
    open($companion_build_lock, '>>', "$build_dir/.latexmk.lock")
        or die "Cannot open build lock: $!";
    unless (flock($companion_build_lock, LOCK_EX | LOCK_NB)) {
        print "Waiting for the other manuscript build to finish...\n";
        flock($companion_build_lock, LOCK_EX) or die "Cannot acquire build lock: $!";
    }
}
