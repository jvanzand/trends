"""
Command Line Interface
"""

from argparse import ArgumentParser
from runner import run

###############
import system_params as sp
###############

parser = ArgumentParser(description="Perform trend analysis")

parser.add_argument('-sn', '--star_name', type=str, metavar='\b', required=True,
                    help='Name of host star')
parser.add_argument('-ms', '--m_star', type=float, metavar='\b', required=True,
                    help='Stellar mass in units of Jupiter masses')
parser.add_argument('-ds', '--d_star', type=float, metavar='\b', required=True,
                    help='Distance to the host star in AU')
parser.add_argument('-gd', '--gdot', type=float, metavar='\b', required=True,
                    help='Linear trend in RVs in m/s/day')
parser.add_argument('-gde', '--gdot_err', type=float, metavar='\b', required=True,
                    help='Error on gamma_dot')
parser.add_argument('-gdd', '--gddot', type=float, metavar='\b', required=True,
                    help='Curvature in RVs in m/s/day/day')
parser.add_argument('-gdde', '--gddot_err', type=float, metavar='\b', required=True,
                    help='Error on gamma_ddot')
parser.add_argument('-bl', '--baseline', type=float, metavar='\b', required=True,
                    help='Length of RV time baseline in days')
parser.add_argument('-rvep', '--rv_epoch', type=float, metavar='\b', required=True,
                    help='Epoch of RV timeseries in BJD, usually around baseline midpoint')
parser.add_argument('-dmu', '--delta_mu', type=float, metavar='\b', required=False,
                    help='Change in astrometric proper motion in milli-arcseconds/yr')
parser.add_argument('-dmue', '--delta_mu_err', type=float, metavar='', required=False,
                    help='Error on dmu')
parser.add_argument('-n', '--num_points', type=int, metavar='\b', required=False,
                    help='Number of orbit models to run')
parser.add_argument('-gn', '--grid_num', type=int, metavar='\b', required=False,
                    help='Dimension of binned probability array')

# # The 'save' and 'plot' args are special. Rather than having an assigned type, the "action='store_true'" argument assumes 1) boolean type and 2) default = False. It also allows there to be no argument after the flag.
parser.add_argument('-s', '--save', action='store_true', required=False,
                    help='Whether to save posterior files')
parser.add_argument('-p', '--plot', action='store_true', required=False,
                    help='Whether to plot joint (a,m) posterior')
parser.add_argument('-r', '--read', type=str, metavar='\b', required=False,
                    help='File path to read in already-calculated posterior array')

args = parser.parse_args()

if __name__=="__main__":
    run(args.star_name, args.m_star, args.d_star, args.gdot, args.gdot_err, 
        args.gddot, args.gddot_err, args.baseline, 
        args.rv_epoch, args.delta_mu, args.delta_mu_err, 
        num_points=args.num_points, grid_num=args.grid_num, 
        save=args.save, plot=args.plot, read_file=args.read)