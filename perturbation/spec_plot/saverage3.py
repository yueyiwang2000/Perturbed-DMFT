#!/usr/bin/env python
# @Copyright 2007 Kristjan Haule
import sys, re, os
import argparse
import numpy as np

nv = list(map(int, np.__version__.split('.')))
if (nv[0], nv[1]) < (1, 6):
    loadtxt = np.loadtxt
    def savetxt(filename, data):
        np.savetxt(filename, data, fmt='%.16g')

if __name__ == '__main__':
    """ Takes several self-energy files and produces an average over these self-energy files """
    usage = """usage: %prog [ options ] argumens

    The script takes several self-energy files and produces an average self-energy

    arguments  -- all input self-energies
    option -o  -- output self-energy file
    """

    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument("-o", "--osig", dest="osig", default='sig.inpx', help="filename of the output self-energy file. Default: 'sig.inp'")
    parser.add_argument("-l", "--lext", dest="m_extn", default='', help="For magnetic calculation, it can be 'dn'.")
    parser.add_argument("-n", "--nexecute", dest="nexecute", action="store_true", default=False, help="Do not execute the comments")
    parser.add_argument("-s", dest="stdev", action='store_true', default=False, help="Computes also the standard deviation - the error of self-energy")
    parser.add_argument('files', nargs='*', help='Self-energy files')

    options = parser.parse_args()

    print('files to average over:', options.files)
    print('output:', options.osig)
    ws_oo = []
    wEdc = []
    wdata = []

    for f in options.files:
        with open(f, 'r') as file:
            dat = file.readlines()
            s_oo = None
            Edc = None
            data = []
            # for line in dat:
            #     m = re.search('#(.*)', line)
            #     if m is not None:
            #         if not options.nexecute:
            #             exec(m.group(1).strip())
            #     else:
            #         data.append(list(map(float, line.split())))

            for line in dat:
                m = re.search('#(.*)', line)
                if m is not None:
                    if not options.nexecute:
                        try:
                            code = m.group(1).strip().replace(" ", "; ").replace(";;", ";")
                            exec(code)
                        except Exception as e:
                            print(f"Warning: Skipping invalid line due to error: {e}")
                else:
                    data.append(list(map(float, line.split())))



            if s_oo is not None:
                ws_oo.append(s_oo)
            if Edc is not None:
                wEdc.append(Edc)
            wdata.append(data)

    with open(options.osig, 'w') as fout:
        if ws_oo:
            ws_oo = np.array(ws_oo)
            as_oo = [sum(ws_oo[:, i]) / len(ws_oo) for i in range(ws_oo.shape[1])]
            print('s_oo=', as_oo)
            print('# s_oo=', as_oo, file=fout)

        if wEdc:
            wEdc = np.array(wEdc)
            aEdc = [sum(wEdc[:, i]) / len(wEdc) for i in range(wEdc.shape[1])]
            print('Edc=', aEdc)
            print('# Edc=', aEdc, file=fout)

        wdata = np.array(wdata)
        wres = np.zeros(wdata.shape[1:], dtype=float)
        for i in range(len(wdata)):
            wres[:, :] += wdata[i, :, :]
        wres *= 1. / len(wdata)

        if options.stdev:
            sw = wdata.shape
            wstd = np.zeros((sw[1], sw[2] - 1), dtype=float)  # no frequency
            for i in range(len(wdata)):
                wstd[:, :] += wdata[i, :, 1:] ** 2
            wstd *= 1. / len(wdata)
            wstd[:, :] = np.sqrt(wstd[:, :] - wres[:, 1:] ** 2)
            wres = np.hstack((wres, wstd))
        
        np.savetxt(fout, wres)
