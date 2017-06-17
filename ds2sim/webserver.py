import time
import json
import signal
import tornado.auth
import tornado.websocket
import tornado.httpserver
import numpy as np
import ds2server.ds2logger
import ds2server.rendering

from tornado.log import enable_pretty_logging


def compileCameraMatrix(right, up, pos):
    """Return serialised camera matrix, or None if `cmat` is invalid.

    """
    # Sanity check `cmat` and construct the forward vector from the right/up
    # vectors.
    try:
        # Unpack the right/up/pos vectors.
        cmat = np.vstack([right, up, pos]).astype(np.float32)
        assert cmat.shape == (3, 3)
        assert not any(np.isnan(cmat.flatten()))
        right, up, pos = cmat[0], cmat[1], cmat[2]

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
        self.logit = ds2server.ds2logger.getLogger('tornado')

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

        # Unpack the camera matrices.
        out = {cname: None for cname in ret}
        ret = {k: v for k, v in ret.items() if v is not None}
        for cname, cdata in ret.items():
            cmat = np.fromstring(cdata, np.float32).reshape(4, 4)
            rot = cmat[:3, :3].tolist()
            right, up = rot[:2]
            pos = cmat[3, :3].tolist()
            out[cname] = {'right': right, 'up': up, 'pos': pos}
        self.write(json.dumps(out).encode('utf8'))


class RestSetCamera(BaseHttp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post(self):
        try:
            payload = self.get_body_argument('data')
        except tornado.web.MissingArgumentError:
            self.logit.warning('Invalid request body')
            self.send_error(400)
            return

        # Parse JSON request.
        try:
            payload = json.loads(payload)
        except json.decoder.JSONDecodeError:
            self.logit.warning('Invalid content in "data" argument')
            self.send_error(400)
            return

        # Select the action according to the command.
        cameras = {}
        try:
            assert isinstance(payload, dict)
            for cname, cdata in payload.items():
                right, up, pos = cdata['right'], cdata['up'], cdata['pos']
                cmat = compileCameraMatrix(right, up, pos)
                assert cmat is not None
                cameras[cname] = cmat
            self.settings['cameras'].update(cameras)
        except (KeyError, AssertionError, TypeError):
            self.logit.warning('Invalid camera matrix')
            self.send_error(400)


class RestRenderScene(BaseHttp):
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
        if cmat is None:
            self.logit.warning(f'Cannot find camera <{camera}>')
            self.send_error(400)
            return

        img = render.renderScene(cmat=cmat, width=width, height=height)
        if img is None:
            self.send_error(400)
        else:
            self.write(img)


class Server:
    def __init__(self, host='127.0.0.1', port=9095, debug=False):
        super().__init__()

        self.host, self.port = host, port
        self.debug = debug
        self._shutdown = False

        # Route Tornado's log messages through our Relays.
        self.logit = ds2server.ds2logger.getLogger('tornado')
        self.logit.info('Server initialised')

    def sighandler(self, signum, frame):
        """ Set the _shutdown flag.

        See `signal module <https://docs.python.org/3/library/signal.html>`_
        for the specific meaning of the arguments.
        """
        msg = 'WebAPI intercepted signal {}'.format(signum)
        self.logit.info(msg)
        self._shutdown = True

    def checkShutdown(self):
        """Initiate shutdown if the _shutdown flag is set."""
        if self._shutdown:
            self.logit.info('WebAPI initiated shut down')
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
            'renderer': ds2server.rendering.getEngine(True),
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
