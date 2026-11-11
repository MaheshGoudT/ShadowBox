import numpy as np
import pytest

from shadowbox.integrations.policy_chunks import ActionChunkController, HttpChunkPolicy, benchmark


def policy(actions):
    return lambda request: dict(schema=request['schema'], actions=actions)


def test_rechecks_mask_and_discards_stale_chunk():
    controller = ActionChunkController(policy([3, 4, 5]))
    mask = np.ones(10, dtype=bool)
    assert controller.action(np.zeros(12), mask) == 3
    mask[4] = False
    assert controller.action(np.zeros(12), mask) == 0
    assert not controller.pending and controller.masked == 1
    controller.action(np.zeros(12), mask)
    assert controller.calls == 2
    controller.reset()
    assert not controller.pending and controller.calls == 0


@pytest.mark.parametrize('actions', [[[0.1] * 7], [0.5], [True], [6], [], [0] * 9])
def test_rejects_arm_actions_and_invalid_chunks(actions):
    controller = ActionChunkController(policy(actions))
    with pytest.raises(ValueError, match='tactical indices'):
        controller.action(np.zeros(12), np.ones(10))


def test_closed_loop_finishes_episode_and_closes_env():
    class Env:
        closed = False
        steps = 0

        def reset(self, seed):
            return np.zeros(12), {'action_mask': np.ones(10)}

        def step(self, action):
            assert action == 0
            self.steps += 1
            return np.ones(12), 1, self.steps == 2, False, {'action_mask': np.ones(10)}

        def close(self):
            self.closed = True

    env = Env()
    controller = ActionChunkController(policy([0] * 4))
    report = benchmark(env, controller)
    assert report['steps'] == 2 and report['policy_calls'] == 1
    assert env.closed and not controller.pending


def test_http_transport_and_decoder_integration():
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            body = json.dumps(dict(schema=request['schema'], actions=[0, 3])).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        controller = ActionChunkController(HttpChunkPolicy(f'http://127.0.0.1:{server.server_port}/predict'))
        assert controller.action(np.zeros(12), np.ones(10)) == 0
        assert controller.action(np.zeros(12), np.ones(10)) == 3
        assert len(requests) == 1
        assert len(requests[0]['observation']) == 12
        assert not requests[0]['action_mask'][6]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
