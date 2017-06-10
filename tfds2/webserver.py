import time
import json
import signal
import logging
import multiprocessing
import tornado.auth
import tornado.websocket
import tornado.httpserver
import numpy as np
import tfds2.rendering

from tornado.log import enable_pretty_logging


def getLogger():
    """Return a custom logger object for DS2 functionality."""
    # Return the logger object if it has handler (means we have already
    # configured it).
    logit = logging.getLogger('tornado')
    if logit.hasHandlers():
        return logit

    # Specify the logging format.
    fmt = (
        '%(asctime)s|%(levelname)s|%(pathname)s:%(lineno)d:%(funcName)s'
        ' (%(process)d)|%(message)s'
    )
    formatter = logging.Formatter(fmt)

    # Configure our own handler (will send the log messages to Relays) and
    # attach it to the logger object.
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logit.addHandler(handler)
    logit.setLevel(logging.DEBUG)
    return logit


def compileCameraMatrix(cmat: list):
    """Return serialised camera matrix, or None if `cmat` is invalid.

    The input `cmat` must be flat list of 9 values. The first three are the
    'right' vector, the next three the 'up' vector, and the last three the
    camera position.
    """
    # Sanity check `cmat` and construct the forward vector from the right/up
    # vectors.
    try:
        # Unpack the right/up/pos vectors.
        cmat = np.array(cmat, np.float32)
        assert cmat.shape == (9, )
        right, up, pos = cmat[0:3], cmat[3:6], cmat[6:9]

        # Ensure righ/up are unit vectors.
        assert (np.linalg.norm(right) - 1) < 1E-5, 'RIGHT is not a unit vector'
        assert (np.linalg.norm(up) - 1) < 1E-5, 'UP is not a unit vector'

        # Ensure right/up vectors are orthogonal.
        eps = np.amax(np.abs(right @ up))
        assert eps < 1E-5, 'Camera vectors not orthogonal'
    except (AssertionError, ValueError):
        return None

    # Compute forward vector and assemble the rotation matrix.
    forward = np.cross(right, up)
    rot = np.vstack([right, up, forward])

    ret = np.eye(4)
    ret[:3, :3] = rot
    ret[3, :3] = pos
    ret = ret.astype(np.float32)
    return ret.flatten('C').tobytes()


class BaseHttp(tornado.web.RequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit = getLogger()

    def write_error(self, status_code, **kwargs):
        if status_code in {404, 405}:
            msg = '{}: Location {} does not exist'
            msg = msg.format(status_code, self.request.uri)
        else:
            msg = '{} Error'.format(status_code)
        self.write(msg)


class RestGetCamera(BaseHttp):
    """API to ping server or update a camera."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post(self):
        # Sanity check: request body must have a 'cmd' and 'data' field.
        try:
            payload = self.get_body_argument('data')
        except tornado.web.MissingArgumentError:
            self.logit.info('invalid request body')
            self.send_error(400)
            return

        # Parse JSON request.
        try:
            payload = json.loads(payload)
        except json.decoder.JSONDecodeError:
            self.logit.info('Invalid content in "data" argument')
            self.send_error(400)
            return

        # Select the action according to the command.
        cameras = self.settings['cameras']
        if payload is None:
            ret = cameras
        else:
            ret = {cname: cameras.get(cname, None) for cname in payload}
        self.write(json.dumps(ret).encode('utf8'))


class RestSetCamera(BaseHttp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post(self):
        try:
            payload = self.get_body_argument('data')
        except tornado.web.MissingArgumentError:
            self.logit.info('Invalid request body')
            self.send_error(400)
            return

        # Parse JSON request.
        try:
            payload = json.loads(payload)
        except json.decoder.JSONDecodeError:
            self.logit.info('Invalid content in "data" argument')
            self.send_error(400)
            return

        # Select the action according to the command.
        self.settings['cameras'].update(payload)


class RestRenderScene(BaseHttp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post(self):
#        print('check1', self.get_body_arguments())
        try:
            payload = self.get_body_argument('data')
        except tornado.web.MissingArgumentError:
            self.logit.info('Invalid request body')
            self.send_error(400)
            return

        # Parse JSON request.
        print('check2', payload)
        try:
            payload = json.loads(payload)
        except json.decoder.JSONDecodeError:
            self.logit.info('Invalid content in "data" argument')
            self.send_error(400)
            return

        try:
            width = int(payload['width'])
            height = int(payload['height'])
            camera = str(payload['camera'])
        except (KeyError, ValueError):
            self.logit.info('Invalid content in "data" argument')
            self.send_error(400)

        # Select the action according to the command.
        render = self.settings['renderer']
        cmat = self.settings['cameras'].get(camera, None)
        img = render.renderScene(cmat=cmat, width=width, height=height)

        if img is None:
            self.send_error(400)
        else:
            self.write(img)


class Server(multiprocessing.Process):
    def __init__(self, host='127.0.0.1', port=9095, debug=False):
        super().__init__()

        self.host, self.port = host, port
        self.debug = debug
        self._shutdown = multiprocessing.Event()

        # Route Tornado's log messages through our Relays.
        self.logit = getLogger()
        self.logit.info('Server initialised')

    def sighandler(self, signum, frame):
        """ Set the _shutdown flag.

        See `signal module <https://docs.python.org/3/library/signal.html>`_
        for the specific meaning of the arguments.
        """
        msg = 'WebAPI intercepted signal {}'.format(signum)
        self.logit.info(msg)
        self._shutdown.set()

    def checkShutdown(self):
        """Initiate shutdown if the _shutdown flag is set."""
        if not self._shutdown.is_set():
            return
        self.logit.info('WebAPI initiated shut down')

        # Stop the serving.
        self.http.stop()

        # Give server some time to process pending events, then stop it.
        time.sleep(1)
        self.http.io_loop.stop()

    def run(self):
        # Install the signal handler to facilitate a clean shutdown.
        signal.signal(signal.SIGINT, self.sighandler)

        # Must not be a daemon because we may spawn sub-processes.
        self.daemon = False
        time.sleep(0.02)

        if self.debug:
            enable_pretty_logging()

        # Initialise the list of Tornado handlers.
        handlers = []
        handlers.append(('/get-camera', RestGetCamera))
        handlers.append(('/set-camera', RestSetCamera))
        handlers.append(('/get-render', RestRenderScene))

        # DS2 parameters that are relevant for the handlers as well.
        settings = {
            'debug': self.debug,
            'cameras': {},
            'renderer': tfds2.rendering.Engine(),
        }

        # Install the handlers and create the Tornado instance.
        app = tornado.web.Application(handlers, **settings)
        self.http = tornado.httpserver.HTTPServer(app)

        # Specify the server port and start the ioloop.
        self.http.listen(self.port, address=self.host)
        tornado_app = tornado.ioloop.IOLoop.current()

        # Periodically check if we should shut down.
        tornado.ioloop.PeriodicCallback(self.checkShutdown, 500).start()

        # Start Tornado event loop.
        print(f'Web Server live at: http://{self.host}:{self.port}')
        self.logit.info('Starting WebAPI')
        tornado_app.start()
        self.logit.info('WebAPI shut down complete')
