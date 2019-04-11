##################################################
# Title:    Observation Data Query               #
# Author:   Dillon Breen                         #
# Institue: Dublin City University               #
##################################################

# Import modules

from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
import astropy.units as u
import warnings
import time
import progressbar

# Self-defined functions

def fov(r, theta, h):

    # Set radius for field of view
    radius = r * np.tan(np.radians(theta/2))

    # Degree for field of view equation
    return np.round(2 * np.degrees(np.arctan(radius/h)), 6)

def quotes(s):

    # Add single quotations to query inputs to proceed
    return "%s" % s

# Gaia archive query

def query(x, ra, dec, gmin, gmax, pmin, pmax):

    # Suppress Warnings
    warnings.filterwarnings("ignore")

    # Query the appropriate Gaia dr2 data values
    job = Gaia.launch_job_async("SELECT * FROM gaiadr2.gaia_source WHERE CONTAINS(POINT('ICRS', gaiadr2.gaia_source.ra, gaiadr2.gaia_source.dec), CIRCLE('ICRS', " + quotes(ra) + ", " + quotes(dec) + ", " + quotes(x) + "))=1 AND phot_g_mean_mag BETWEEN " + quotes(gmin) + " AND " + quotes(gmax) + " AND parallax BETWEEN " + quotes(pmin) + " AND " + quotes(pmax) + "", background=True)

    # It is possible to change Gaia archive query to alternative instrument above (astroquery module)

    # Place query into data frame
    new = pd.DataFrame()
    result = job.get_results().to_pandas()
    data = new.append(result)

    # Generate data from host server (Gaia)
    return data

# Data generation

def gen_data(interval, r, degree, vmin, vmax, ra, dec, file, stellar_type):

    # Upper & lower parallax limits (30000 pc upper limit approximation radius of Milk Way)
    if file == ("SgrA-Type" + quotes(stellar_type) + ".csv"):
        values = np.arange(interval, 8000 + interval, interval)
        difference = interval
    else:
        values = np.arange(interval, r + interval, interval)
        difference = interval

    # Create data frame
    stars = pd.DataFrame()

    # Input galactic cross-section parameters
    for i, j in zip(fov(r, degree, values), values):

        # Parallax limitations
        m = np.round((np.multiply(j, u.pc).to(u.mas, equivalencies=u.parallax())).value, 6)
        n = np.round((np.multiply((j + difference), u.pc).to(u.mas, equivalencies=u.parallax())).value, 6)

        # Magnitude limitations
        v = np.round(vmin + 5 * np.log10(j/10), 6)
        w = np.round(vmax + 5 * np.log10(j/10), 6)

        # Display the assigned parameters of each query
        print("\nDegree: " + quotes(i) + " Lower parallax: " + quotes(n) + " Upper parallax: " + quotes(m) + "\nLower apparent magnitude: " + quotes(v) + " Upper apparent magnitude: " + quotes(w))

        # Query host data
        result = query(i, ra, dec, v, w, n, m)

        # Append to data frame
        stars = stars.append(result, ignore_index=True)

    # Write to file
    stars.to_csv(file, mode="a")

    # SgrA : Observation in the direction towards the galactic center (SgrA*)
    # OppSgrA : Observation in the direction away from galactic center (SgrA*)

    # Keep track of query progression
    print("\n Stellar type " + stellar_type + " complete \n")

    # Stellar type limited query
    return

# Query call

def main():

    # Input stellar type
    types = input("Enter the desired stellar type (O - M, or all): ")

    if types in ["O", "B", "A", "F", "G", "K", "M", "all"]:

        # Input query parameters
        interval = float(input("Enter the desired scan interval (parsec): "))
        r = float(input("Enter the desired outer observation range (parsec): "))
        degree = float(input("Enter the desired scan angle (at furtherest point): "))

        # Galactic center as focus
        ra, dec, opp_ra, opp_dec = 266.41683, -29.00781, 86.41683, 29.00781

        # Specify stellar type
        if types == "O":
            print("\nQuerying stellar type O.")
            gen_data(interval, r, degree, -4.5, -3.6, ra, dec, "SgrA-TypeO.csv", "O")
            gen_data(interval, r, degree, -4.5, -3.6, opp_ra, opp_dec, "OppSgrA-TypeO.csv", "O")
            print("\n Stellar type O query complete.")

        if types == "B":
            print("\nQuerying stellar type B.")
            gen_data(interval, r, degree, -3.3, 1.1, ra, dec, "SgrA-TypeB.csv", "B")
            gen_data(interval, r, degree, -3.3, 1.1, opp_ra, opp_dec, "OppSgrA-TypeB.csv", "B")
            print("\n Stellar type B query complete.")

        if types == "A":
            print("\nQuerying stellar type A.")
            gen_data(interval, r, degree, 1.5, 2.4, ra, dec, "SgrA-TypeA.csv", "A")
            gen_data(interval, r, degree, 1.5, 2.4, opp_ra, opp_dec, "OppSgrA-TypeA.csv", "A")
            print("\n Stellar type A query complete.")

        if types == "F":
            print("\nQuerying stellar type F.")
            gen_data(interval, r, degree, 3.0, 4.4, ra, dec, "SgrA-TypeF.csv", "F")
            gen_data(interval, r, degree, 3.0, 4.4, opp_ra, opp_dec, "OppSgrA-TypeF.csv", "F")
            print("\n Stellar type F query complete.")

        if types == "G":
            print("\nQuerying stellar type G.")
            gen_data(interval, r, degree, 4.7, 5.6, ra, dec, "SgrA-TypeG.csv", "G")
            gen_data(interval, r, degree, 4.7, 5.6, opp_ra, opp_dec, "OppSgrA-TypeG.csv", "G")
            print("\n Stellar type G query complete.")

        if types == "K":
            print("\nQuerying stellar type K.")
            gen_data(interval, r, degree, 6.0, 8.1, ra, dec, "SgrA-TypeK.csv", "K")
            gen_data(interval, r, degree, 6.0, 8.1, opp_ra, opp_dec, "OppSgrA-TypeK.csv", "K")
            print("\n Stellar type K query complete.")

        if types == "M":
            print("\nQuerying stellar type M.")
            gen_data(interval, r, degree, 8.7, 14.4, ra, dec, "SgrA-TypeM.csv", "M")
            gen_data(interval, r, degree, 8.7, 14.4, opp_ra, opp_dec, "OppSgrA-TypeM.csv", "M")
            print("\n Stellar type M query complete.")

        elif types == "all":
            print("\nQuerying all available stellar types. \nQuerying stellar type O.")
            gen_data(interval, r, degree, -4.5, -3.6, ra, dec, "SgrA-TypeO.csv", "O")
            gen_data(interval, r, degree, -4.5, -3.6, opp_ra, opp_dec, "OppSgrA-TypeO.csv", "O")
            print("\n Stellar type O query complete. \nQuerying stellar type B")
            gen_data(interval, r, degree, -3.3, 1.1, ra, dec, "SgrA-TypeB.csv", "B")
            gen_data(interval, r, degree, -3.3, 1.1, opp_ra, opp_dec, "OppSgrA-TypeB.csv", "B")
            print("\n Stellar type B query complete. \nQuerying stellar type A")
            gen_data(interval, r, degree, 1.5, 2.4, ra, dec, "SgrA-TypeA.csv", "A")
            gen_data(interval, r, degree, 1.5, 2.4, opp_ra, opp_dec, "OppSgrA-TypeA.csv", "A")
            print("\n Stellar type A query complete. \nQuerying stellar type F")
            gen_data(interval, r, degree, 3.0, 4.4, ra, dec, "SgrA-TypeF.csv", "F")
            gen_data(interval, r, degree, 3.0, 4.4, opp_ra, opp_dec, "OppSgrA-TypeF.csv", "F")
            print("\n Stellar type F query complete. \nQuerying stellar type G")
            gen_data(interval, r, degree, 4.7, 5.6, ra, dec, "SgrA-TypeG.csv", "G")
            gen_data(interval, r, degree, 4.7, 5.6, opp_ra, opp_dec, "OppSgrA-TypeG.csv", "G")
            print("\n Stellar type G query complete. \nQuerying stellar type K")
            gen_data(interval, r, degree, 6.0, 8.1, ra, dec, "SgrA-TypeK.csv", "K")
            gen_data(interval, r, degree, 6.0, 8.1, opp_ra, opp_dec, "OppSgrA-TypeK.csv", "K")
            print("\n Stellar type K query complete. \nQuerying stellar type M")
            gen_data(interval, r, degree, 8.7, 14.4, ra, dec, "SgrA-TypeM.csv", "M")
            gen_data(interval, r, degree, 8.7, 14.4, opp_ra, opp_dec, "OppSgrA-TypeM.csv", "M")
            print("\n Stellar type M query complete. \n All queries complete.")

        else:
            print("Not a valid option, try again.")
            main()

    # Call stellar type dependent query
    return

main()
