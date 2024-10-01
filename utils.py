import argparse
import logging
import matplotlib


__parser = argparse.ArgumentParser(prog='Population Model')
__parser.add_argument('--gui_backend', default='TkAgg', dest='gui_backend',
                      choices=['QtAgg', 'TkAgg', 'gtk3cairo', 'gtk4agg', 'macosx', 'gtk4cairo', 'qt5agg'])
__parser.add_argument('--log', default='debug', dest='log',
                      choices=['debug', 'info', 'error'])

logging.basicConfig(format="%(asctime)s [%(levelname)s]  %(message)s")
logger = logging.getLogger("main")


def parse_args():
    print('Parsing command line arguments...')
    args = __parser.parse_args()

    if args.log is not None:
        log = args.log.lower()
        level = logging.INFO if log == 'info' else (logging.DEBUG if log == 'debug' else logging.ERROR)
        logger.setLevel(level=level)

    if args.gui_backend is not None:
        print(f'USING matplotlib_gui_backend = {args.gui_backend}')
        try:
            matplotlib.use(args.gui_backend.lower())
        except Exception as e:
            print(f'ERROR {e}')
            print(f'USING matplotlib_gui_backend = TkAgg (default)')

    print()
