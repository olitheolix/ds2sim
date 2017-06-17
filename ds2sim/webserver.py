import io
import time
import json
import signal
import tornado.auth
import tornado.websocket
import tornado.httpserver
import ds2sim.horde
import ds2sim.camera
import ds2sim.logger

import numpy as np
import PIL.Image as Image
from tornado.log import enable_pretty_logging


def rawToJpeg(img_mat, quality: int):
    """ Return JPEG encoded binary image data.

    Return None if there is an error.

    Args:
        img_mat (NumPy uint8): a Numpy array.
        quality (int): compression level [0-100]

    Returns:
        bytes: the binary image, or None if an error occurred.
    """
    # Convert the image to JPEG and return it.
    try:
        assert isinstance(img_mat, np.ndarray)
        assert img_mat.dtype == np.uint8
        assert np.prod(img_mat.shape) > 0
        assert isinstance(quality, int)
        img = Image.fromarray(img_mat).convert('RGB')
        quality = int(np.clip(quality, 0, 100))
    except (KeyError, AttributeError, AssertionError):
        return None

    # Save the Image as JPEG and return.
    buf = io.BytesIO()
    img.save(buf, 'jpeg', quality=quality)
    buf.seek(0)
    return buf.read()


class BaseHttp(tornado.web.RequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit = ds2sim.logger.getLogger('tornado')

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
                cmat = ds2sim.camera.compileCameraMatrix(right, up, pos)
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
        img = rawToJpeg(img, quality=90)
        if img is None:
            self.send_error(400)
        else:
            self.write(img)


class Server:
    def __init__(
            self, host='127.0.0.1', port=9095,
            default_scene=False, debug=False):
        super().__init__()

        self.host, self.port = host, port
        self.debug = debug
        self.default_scene = default_scene
        self._shutdown = False

        # Route Tornado's log messages through our Relays.
        self.logit = ds2sim.logger.getLogger('tornado')
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

        # Instantiate Horde, and load the default scene, if requested.
        horde = ds2sim.horde.Engine(512, 512)
        if self.default_scene:
            horde._loadDemoScene(num_cubes=200, seed=0)
        settings = {
            'debug': self.debug,
            'cameras': {},
            'renderer': horde,
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
