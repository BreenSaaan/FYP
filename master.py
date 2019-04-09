##################################################
# Title:    Observation Data Processing/Analysis #
# Author:   Dillon Breen                         #
# Institue: Dublin City University               #
##################################################

# Import modules

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pylab as plt

# Self-defined functions

def gc(ra, dec, p, pmra, pmdec, rv):

    # Coordinate transformation (ICRS to Galactic)
    coordinates = coord.ICRS(ra=np.multiply(ra, u.degree), dec=np.multiply(dec, u.degree), distance=(list(p.values) * u.mas).to(u.pc, u.parallax()), pm_ra_cosdec=np.multiply(pmra, u.mas/u.yr), pm_dec=np.multiply(pmdec, u.mas/u.yr), radial_velocity=np.multiply(rv, u.km/u.s))


    # Returns (x, y, z) spatial coordinates wrt galactic center, thus distance is the square difference
    # Returns (v_x, v_y, v_z) velocity coordinates wrt galactic center, thus velocity is the square difference

    # Galactocentric coordinates
    return coordinates.transform_to(coord.Galactocentric(galcen_distance=8 * u.kpc, z_sun=0 * u.pc))

def fix_points(n, x, y, xf, yf) :

    # Setting fixed points
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)

    # Polynomial function with fixed points
    return params[:n + 1]

def quotes(s):

    # Add single quotations to query inputs (file name & radius)
    return "%s" % s

def ss_model():

    # Planet names
    planet = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

    # Planet distances in AU
    distance = [0.39, 0.723, 1, 1.524, 5.203, 9.539, 19.18, 30.06]

    # Planet velocities
    velocity = [47.36, 35.02, 29.78, 24.077, 13.07, 9.69, 6.81, 5.43]

    # Append values to data frame
    df = pd.DataFrame(zip(planet, distance, velocity), columns=["Planet", "Distance", "Velocity"])

# Predicted Keplerian model
    g = 4.30091E-3
    xc = np.linspace(0.01, 35, 100)
    yc = np.sqrt((g * 1.0014) / (4.84814E-6 * xc))
    plt.plot(xc, yc, color="blue", label="Keplerian Model", zorder=1)

    # Assign name of each planet to appropriate data point
    for i, j in enumerate(planet):
        x = df["Distance"][i]
        y = df["Velocity"][i]
        plt.scatter(x, y, marker="o", s=10, color="black", zorder=2)
        plt.text(x + 0.5, y + 0.5, j, fontsize=10, zorder=3)

    # Plot characteristics
    plt.title("Solar System Rotational Curves")
    plt.xlabel("Solar System Radius ($AU$)")
    plt.ylabel("Rotational Velocity ($kms^{-1}$)")
    plt.xlim(0, 35)
    plt.ylim(0, 55)
    plt.legend(loc="upper right")
    plt.show()

    # solar system model
    return

# Data generation

def gen_data(stellar_type):

    # Read observational data
    df1 = pd.read_csv("SgrA-Type" + stellar_type + ".csv", usecols=["ra", "ra_error", "dec", "dec_error", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "radial_velocity", "radial_velocity_error"], sep=',', low_memory=False, error_bad_lines=False).astype(float)

    df2 = pd.read_csv("OppSgrA-Type" + stellar_type + ".csv", usecols=["ra", "ra_error", "dec", "dec_error", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "radial_velocity", "radial_velocity_error"], sep=',', low_memory=False, error_bad_lines=False).astype(float)


    # Calculation for distance of sources towards SgrA*
    df1["x"] = gc(df1["ra"], df1["dec"], df1["parallax"], df1["pmra"], df1["pmdec"], df1["radial_velocity"]).x.value
    df1["y"] = gc(df1["ra"], df1["dec"], df1["parallax"], df1["pmra"], df1["pmdec"], df1["radial_velocity"]).y.value
    df1["parsec"] = np.sqrt((df1["x"] ** 2) + (df1["y"] ** 2))
    df1["x_error"] = gc(df1["ra_error"], df1["dec_error"], df1["parallax_error"], df1["pmra_error"], df1["pmdec_error"], df1["radial_velocity_error"]).x.value
    df1["y_error"] = gc(df1["ra_error"], df1["dec_error"], df1["parallax_error"], df1["pmra_error"], df1["pmdec_error"], df1["radial_velocity_error"]).y.value
    df1["parsec_error"] = np.sqrt((df1["x_error"] ** 2) + (df1["y_error"] ** 2))

    # Calculation for distances of sources away from SgrA*
    df2["x"] = gc(df2["ra"], df2["dec"], df2["parallax"], df2["pmra"], df2["pmdec"], df2["radial_velocity"]).x.value
    df2["y"] = gc(df2["ra"], df2["dec"], df2["parallax"], df2["pmra"], df2["pmdec"], df2["radial_velocity"]).y.value
    df2["parsec"] = np.sqrt((df2["x"] ** 2) + (df2["y"] ** 2))
    df2["x_error"] = gc(df2["ra_error"], df2["dec_error"], df2["parallax_error"], df2["pmra_error"], df2["pmdec_error"], df2["radial_velocity_error"]).x.value
    df2["y_error"] = gc(df2["ra_error"], df2["dec_error"], df2["parallax_error"], df2["pmra_error"], df2["pmdec_error"], df2["radial_velocity_error"]).y.value
    df2["parsec_error"] = np.sqrt((df2["x_error"] ** 2) + (df2["y_error"] ** 2))

    # Merge the data frames
    df = pd.concat([df1, df2], sort=True, ignore_index=True)

    # Parsecs data frame to kiloparsecs data frame
    df["kiloparsec"] = ((list(df["parsec"]) * u.pc).to(u.kpc)).value
    df["kiloparsec_error"] = ((list(df["parsec_error"]) * u.pc).to(u.kpc)).value

    # Remove nan values from radial velocity column
    df = df[np.isfinite(df["radial_velocity"])]

    # Calculation of rotational velocity of sources
    df["v_x"] = gc(df["ra"], df["dec"], df["parallax"], df["pmra"], df["pmdec"], df["radial_velocity"]).v_x.value
    df["v_y"] = gc(df["ra"], df["dec"], df["parallax"], df["pmra"], df["pmdec"], df["radial_velocity"]).v_y.value
    df["v_z"] = gc(df["ra"], df["dec"], df["parallax"], df["pmra"], df["pmdec"], df["radial_velocity"]).v_z.value
    df["rotational_velocity"] = np.sqrt((df["v_x"] ** 2) + (df["v_y"] ** 2) + (df["v_z"] ** 2))
    df["v_x_error"] = gc(df["ra_error"], df["dec_error"], df["parallax_error"], df["pmra_error"], df["pmdec_error"], df["radial_velocity_error"]).v_x.value
    df["v_y_error"] = gc(df["ra_error"], df["dec_error"], df["parallax_error"], df["pmra_error"], df["pmdec_error"], df["radial_velocity_error"]).v_y.value
    df["v_z_error"] = gc(df["ra_error"], df["dec_error"], df["parallax_error"], df["pmra_error"], df["pmdec_error"], df["radial_velocity_error"]).v_z.value
    df["rotational_velocity_error"] = np.sqrt((df["v_x_error"] ** 2) + (df["v_y_error"] ** 2) + (df["v_z_error"] ** 2))

    # Parsecs data frame to meters data frame
    df["meter"] = ((list(df["parsec"]) * u.pc).to(u.m)).value
    df["meter_error"] = ((list(df["parsec_error"]) * u.pc).to(u.m)).value

    # km/s data frame to m/s data frame
    df["rotational_velocity_ms"] = ((list(df["rotational_velocity"]) * u.km/u.s).to(u.m/u.s)).value
    df["rotational_velocity_ms_error"] = ((list(df["rotational_velocity_error"]) * u.km/u.s).to(u.m/u.s)).value

    # Mass calculation (M = (R * V ** 2)/G)
    df["mass"] = (df["meter"] * (df["rotational_velocity_ms"]) ** 2)/6.67408E-11

    # Mass error calculation (DM = |R * V^2| * sqrt((DR/R) ** 2) + (|2| * V * DV/V ** 2) ** 2)
    df["rotational_velocity_squared_error"] = np.abs(2) * df["rotational_velocity_ms"] * df["rotational_velocity_ms_error"]
    df["radius_times_velocity_squared"] = np.abs(df['meter'] * (df["rotational_velocity_ms"] ** 2))
    df["radius_error_over_radius_squared"] = (df["meter_error"]/df["meter"]) ** 2
    df["rotational_velocity_squared_error_over_velocity_squared_squared"] = (df["rotational_velocity_squared_error"]/(df["rotational_velocity_ms"] ** 2)) ** 2
    df["mass_error"] = (df["radius_times_velocity_squared"] * np.sqrt(df["radius_error_over_radius_squared"] + df["rotational_velocity_squared_error_over_velocity_squared_squared"]))/6.67408E-11

    # Convert to mass to solar mass
    df["mass"] = ((list(df["mass"]) * u.kg).to(u.M_sun)).value
    df["mass_error"] = ((list(df["mass_error"]) * u.kg).to(u.M_sun)).value

    # Reset data frame index
    df = df.reset_index(drop=True)

    print(df[["kiloparsec", "kiloparsec_error", "rotational_velocity", "rotational_velocity_error", "mass", "mass_error"]])

    # Generate observational data
    return df

def gen_types(stellar_type, sample):

    # Create data frame to append files
    df = pd.DataFrame()

    # Stellar type selection
    if stellar_type == "O":
        df = df.append(gen_data("O"))
    if stellar_type == "B":
        df = df.append(gen_data("B"))
    if stellar_type == "A":
        df = df.append(gen_data("A"))
    if stellar_type == "F":
        df = df.append(gen_data("F"))
    if stellar_type == "G":
        df = df.append(gen_data("G"))
    if stellar_type == "K":
        df = df.append(gen_data("K"))
    if stellar_type == "M":
        df = df.append(gen_data("M"))
    elif stellar_type == "all":
        df = df.append(gen_data("O"))
        df = df.append(gen_data("B"))
        df = df.append(gen_data("A"))
        df = df.append(gen_data("F"))
        df = df.append(gen_data("G"))
        df = df.append(gen_data("K"))
        df = df.append(gen_data("M"))

    # Sample data
    if sample in range(0, len(df)):
        df = df.sample(sample)

    elif sample == "all":
        print("Total data points included: " + quotes(len(df)) + " sources.")

    else:
        print("Sample input outside possible sample range. \nThere are a total of " + quotes(len(df)) + " data points available.")

    # Generate data for specific stellar type
    return df

def gen_bins(frame):

    # Bin intervals
    bin_range = np.linspace(0, 16, 20)

    # Binning values
    frame["bins"] = pd.cut(frame["kiloparsec"], bin_range)
    frame["bins_error"] = pd.cut(frame["kiloparsec_error"], bin_range)

    # Binned intervals & counts per interval for distribution
    count = frame.groupby("bins")["kiloparsec"].count().reset_index(name="bin_count")
    error = frame.groupby("bins_error")["kiloparsec_error"].count().reset_index(name="count_error")

    # Midpoints for accurately plotting x-axis
    mid = [(a + b)/2 for a,b in zip(bin_range[:-1], bin_range[1:])]
    mid_error = [(a + b)/2 for a, b in zip(bin_range[:-1], bin_range[1:])]
    count["bins"] = mid
    error["bins_error"] = mid_error
    error["bins_error"] = (error["bins_error"] * (frame["parallax_error"]/frame["parallax"])).reset_index(drop=True)
    
    # Set density as arbitrary units
    count["bin_count"] = count["bin_count"]/count["bin_count"].values.max()
    error["count_error"] = error["count_error"]/error["count_error"].values.max()

    # Merge data frames
    frame = pd.merge(count, error, how="left", left_index=True, right_index=True)

    print(frame)

    # Generate bin range for source count
    return frame

# Modelisation

def stellar_distribution_plot(frame, fit):

    # Plot the data and model
    plt.errorbar(frame["bins"], frame["bin_count"], xerr=frame["bins_error"], yerr=frame["count_error"], fmt="o", ms=0.5, color="black", ecolor="grey", elinewidth=0.25, label="Gaia DR2", zorder=1)

    # Density distribution model
    summation_s_0 = frame["bin_count"].values.max()
    l_c, r_b = 3, 2.5
    xb = np.linspace(0, 16, 50)
    yb = (summation_s_0 * l_c)/np.sqrt(((xb - r_b) ** 2) + (l_c ** 2))
    xp = np.array([frame['bins'].iloc[0], frame['bins'].iloc[14], frame['bins'].iloc[-1]])
    yp = np.array([frame['bin_count'].iloc[0], frame['bin_count'].iloc[14], frame['bin_count'].iloc[-1]])
    params = fix_points(fit, xb, yb, xp, yp)
    ppoly = np.polynomial.Polynomial(params)
    pdata = np.linspace(0, 16, 50)
    plt.plot(pdata, ppoly(pdata), ms=1, color="blue", label="Best fit density distribution model", zorder=2)

    # Plot characteristics
    plt.title("Stellar Density Distribution")
    plt.legend(loc="upper right")
    plt.xlabel("Galactic radius ($kpc$)")
    plt.ylabel("Density distribution ($arbitrary$ $units$)")
    plt.xlim(0, 16) # Adjustment for galactic radius
    plt.ylim(0, frame["bin_count"].values.max() + (1/10 * frame["bin_count"].values.max()))
    plt.show()

    # Stellar disc model against observational data (Gaia)
    return

def rotational_curvature_plot(stellar_type, sample, fit):
    
    # Galactic mass prediction (typically between 1E11 & 1E12)
    gal_mass = float(input("Enter a galactic mass prediction (solar masses): "))
    
    if gal mass = float:
        gal_mass = gal_mass
        
    else:
        rotational_curvature_plot(stellar_type, sample, fit)
    
    # Define the data frame
    df = gen_types(stellar_type, sample)

    # Observational data plot
    plt.errorbar(df["kiloparsec"], df["rotational_velocity"], fmt="o", ms=0.5, color="black", xerr=df["kiloparsec_error"], yerr=df["rotational_velocity_error"], elinewidth=0.25, ecolor="grey", label="Gaia DR2", zorder=1)

    # Best fit polynomial
    x = df["kiloparsec"].values
    y = df["rotational_velocity"].values
    xp = np.array([0])
    yp = np.array([0])
    params = fix_points(fit, x, y, xp, yp)
    ppoly = np.polynomial.Polynomial(params)
    pdata = np.linspace(0, 16, 50)
    plt.plot(pdata, ppoly(pdata), ms=2, color="blue", label="Best fit rotation curve", zorder=4)

    # Predicted Keplerian model
    g = 4.30091E-3
    xkm = np.linspace(5.4, 16, 50)
    ykm = np.sqrt((g * gal_mass) / (1000 * xkm))
    xk = np.array([0])
    yk = np.array([0])
    params = fix_points(fit, xkm, ykm, xk, yk)
    kpoly = np.polynomial.Polynomial(params)
    kdata = np.linspace(0, 16, 50)
    plt.plot(kdata, kpoly(kdata), "--", ms=2, color="blue", label="Predicted Keplerian model", zorder=2)

    # Non-Kelperian model
    xnk = pdata
    ynk = ppoly(pdata) - ykm
    xn = np.array([0])
    yn = np.array([0])
    params = fix_points(fit, xnk, ynk, xn, yn)
    npoly = np.polynomial.Polynomial(params)
    ndata = np.linspace(0, 16, 50)
    plt.plot(ndata, npoly(ndata), "--", ms=2, color="red", label="Dark matter contribution", zorder=3)

    # The Sun
    plt.plot(8, 220, marker="o", ms=8, color="yellow", label="Sun", zorder=5)

    # Plot characteristics
    plt.title("Rotational Curvature")
    plt.xlabel("Galactic radius ($kpc$)")
    plt.ylabel("Rotational velocity ($kms^{-1}$)")
    plt.legend(loc="upper right")
    plt.xlim(0, 16)
    plt.ylim(0, 400)
    plt.show()

    # Milky Way rotation curve
    return

def dark_matter_distribution_plot(stellar_type, sample, fit):
    
    # Galactic mass prediction (typically between 1E11 & 1E12)
    gal_mass = float(input("Enter a galactic mass prediction (solar masses): "))
    
    if gal mass = float:
        gal_mass = gal_mass
        
    else:
        dark_matter_distribution_plot(stellar_type, sample, fit)

    # Define the data frame
    df = gen_types(stellar_type, sample)

    # Observational data plot
    plt.errorbar(df["kiloparsec"], df["mass"], fmt="o", ms=0.5, color="black", xerr=df["kiloparsec_error"], yerr=df["mass_error"],ecolor="grey", elinewidth=0.25, label="Gaia DR2", zorder=1)

    # Best fit polynomial
    x = df["kiloparsec"].values
    y = df["mass"].values
    xp = np.array([0])
    yp = np.array([0])
    params = fix_points(fit, x, y, xp, yp)
    poly = np.polynomial.Polynomial(params)
    data = np.linspace(0, 16, 50)
    plt.plot(data, poly(data), color="blue", label="Mass density distribution", zorder=4)

    # Predicted Keplerian model
    g = 4.30091E-3
    xkm = np.linspace(5.4, 16, 50)
    ykm = np.sqrt((g * gal_mass) / (1000 * xkm))
    xk = np.array([0])
    yk = np.array([0])
    params = fix_points(fit, xkm, ykm, xk, yk)
    kpoly = np.polynomial.Polynomial(params)
    kdata = np.linspace(0, 16, 50)
    plt.plot(kdata, kpoly(kdata), "--", ms=2, color="blue", label="Predicted Keplerian model", zorder=2)

    # Non-Kelperian model
    xnk = data
    ynk = poly(data) - ykm
    xn = np.array([0])
    yn = np.array([0])
    params = fix_points(fit, xnk, ynk, xn, yn)
    npoly = np.polynomial.Polynomial(params)
    ndata = np.linspace(0, 16, 50)
    plt.plot(ndata, npoly(ndata), "--", ms=2, color="red", label="Dark matter contribution", zorder=3)

    # The Sun
    sun_mass = (((8000 * u.pc).to(u.m)) * ((220 * u.km / u.s).to(u.m / u.s)) ** 2 / 6.67408E-11)
    plt.plot(8, sun_mass, marker="o", ms=8, color="yellow", label="Sun", zorder=5)

    # Plot characteristics
    plt.title("Dark Matter Distribution")
    plt.xlabel("Galactic radius ($kpc$)")
    plt.ylabel("Mass ($solar$ $mass$)")
    plt.legend(loc="upper right")
    plt.xlim(0, 16)
    plt.ylim(0, 1E12)
    plt.show()

    # Mass distribution model
    return

# Query call

def stellar_distribution(stellar_type, sample, fit):

    print("\nStellar density distribution for " + quotes(stellar_type) + " stellar types. \nSample size: " + quotes(sample) + ".")

    # Assign name to data frame
    Type = gen_types(stellar_type, sample)

    # Stellar distribution model
    stellar_distribution_plot(gen_bins(Type), fit)

    # Plot model and observational data
    return

def rotational_curvature(stellar_type, sample, fit):

    print("\nRotational curvature for " + quotes(stellar_type) + " stellar types. \nSample size: " + quotes(sample) + ".")

    # Rotational curvature
    rotational_curvature_plot(stellar_type, sample, fit)

    # Plot model and observational data
    return

def dark_matter_distribution(stellar_type, sample, fit):

    print("\nMass distribution for " + quotes(stellar_type) + " stellar types. \nSample size: " + quotes(sample) + ".")

    # Plot all data
    dark_matter_distribution_plot(stellar_type, sample, fit)

    # Plot model and observational data
    return

def main():

    # Input stellar type
    stellar_type = input("Enter the desired stellar type (O - M, or all): ")

    if stellar_type in ["O", "B", "A", "F", "G", "K", "M", "all"]:

        sample = input("Enter the desired sample size (numerical quantity or all): ")

        if sample != "all":
            sample = int(sample)

        else:
             sample = "all"

        # stellar_distribution(a, b):
        # a = Sample size or "all", b = Model fit polynomial degree
        stellar_distribution(stellar_type, sample, 4)

        # rotational_curvature(a, b):
        # a = Sample size or "all", b = Model best fit polynomial degree
        rotational_curvature(stellar_type, sample, 7)

        # dark_matter_distribution(a, b):
        # a = Sample size or "all", b = Model best fit polynomial degree
        dark_matter_distribution(stellar_type, sample, 3)

    else:
        print("Not a valid option, try again.")
        main()

    return

main()
