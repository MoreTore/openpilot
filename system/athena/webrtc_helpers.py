import asyncio
import threading
import json
import logging
import queue
import requests

from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription, RTCIceCandidate, RTCRtpCodecCapability, RTCIceServer, RTCStatsReport
from aiortc.sdp import candidate_from_sdp
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack
from openpilot.system.manager.process_config import managed_processes, NativeProcess

def get_ice_servers():
  api_key = "26e6c30483590d95b1f878fc361f8286151e"
  url = f"https://gambitstreamer.metered.live/api/v1/turn/credentials?apiKey={api_key}"

  try:
    resp = requests.get(url)
    if resp.status_code == 200:
      ice_servers = resp.json()
      print(f"[INFO] Fetched TURN server credentials")
      return [RTCIceServer(**server) for server in ice_servers]
    else:
      print(f"[ERROR] Failed to fetch TURN server credentials: {resp.status_code}")
      return []
  except Exception as e:
    print(f"[ERROR] Exception occurred while fetching TURN server credentials: {e}")
    return []

def add_track(camera_type: str, pc: RTCPeerConnection) -> None:
    video_track = LiveStreamVideoStreamTrack(camera_type)
    transceiver = pc.addTransceiver("video", direction="sendonly")
    h264_capability = RTCRtpCodecCapability(
        mimeType="video/H264",
        clockRate=90000,
        parameters={
            "level-asymmetry-allowed": "1",
            "packetization-mode": "1",
            "profile-level-id": "42e01f"
        }
    )
    print(f"Codec: {h264_capability.mimeType}, "
          f"Clock Rate: {h264_capability.clockRate}, "
          f"Parameters: {h264_capability.parameters}")

    transceiver.setCodecPreferences([h264_capability])
    pc.addTrack(video_track)

async def create_offer(pc: RTCPeerConnection):
    try:
        print("creating offer")
        offer: RTCSessionDescription = await pc.createOffer()
        print(f"Created Offer {offer}")
        await pc.setLocalDescription(offer)
        print("Set Local Description")
    except Exception as e:
        print(f"Failed to create offer: {e}")
        await pc.close()

async def set_answer(pc, data):
    try:
        answer = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
        await pc.setRemoteDescription(answer)
        print("Set remote description (answer)")
    except Exception as e:
        print(f"Failed to set answer: {e}")
        await pc.close()

async def set_candidate(pc: RTCPeerConnection, candidate_data):
    print("Called set candidate")
    candidate_sdp = candidate_data["candidate"]
    parsed_candidate: RTCIceCandidate = candidate_from_sdp(candidate_sdp)
    parsed_candidate.sdpMid = candidate_data.get("sdpMid")
    parsed_candidate.sdpMLineIndex = candidate_data.get("sdpMLineIndex")

    await pc.addIceCandidate(parsed_candidate)
    print(f"[INFO] Added ICE candidate {parsed_candidate}")

class Streamer():
    def __init__(self, sdp_send_queue: queue.Queue, sdp_recv_queue: queue.Queue, ice_send_queue: queue.Queue):
        print("Called init")
        self.pc = None
        self.data_channel = None
        self.sdp_send_queue = sdp_send_queue
        self.sdp_recv_queue = sdp_recv_queue
        self.ice_send_queue = ice_send_queue
        self.camera = None
        self.encoder = None
        self.should_stop = False
        self.tracks = {
            "driver": LiveStreamVideoStreamTrack("driver"),
            "road": LiveStreamVideoStreamTrack("road"),
            "wideRoad": LiveStreamVideoStreamTrack("wideRoad"),
        }

    async def renegotiate(self):
        try:
            print("[INFO] Starting renegotiation...")
            await create_offer(self.pc)
            print("[INFO] Created new offer and set local description.")

            # # Filter SDP (optional, if needed for relay or specific candidates)
            # filtered_sdp = "\r\n".join(
            #     line for line in self.pc.localDescription.sdp.splitlines()
            #     if not line.startswith("a=candidate") or "relay" in line
            # ) + "\r\n"

            # Send the new SDP offer over the dataChannel
            if self.data_channel and self.data_channel.readyState == "open":
                message = json.dumps({
                    'type': self.pc.localDescription.type,
                    'sdp': self.pc.localDescription.sdp,
                })
                self.data_channel.send(message)
                print("[INFO] Sent new SDP offer over dataChannel.")
            else:
                print("[ERROR] Data channel is not open. Cannot send SDP offer.")
        except Exception as e:
            print(f"[ERROR] Failed to renegotiate: {e}")

    def add_tracks(self) -> None:
        for track_type, track in self.tracks.items():
            transceiver = self.pc.addTransceiver("video", direction="sendonly")
            h264_capability = RTCRtpCodecCapability(
                mimeType="video/H264",
                clockRate=90000,
                parameters={
                    "level-asymmetry-allowed": "1",
                    "packetization-mode": "1",
                    "profile-level-id": "42e01f"
                }
            )
            print(f"Codec: {h264_capability.mimeType}, "
                f"Clock Rate: {h264_capability.clockRate}, "
                f"Parameters: {h264_capability.parameters}")

            transceiver.setCodecPreferences([h264_capability])
            self.pc.addTrack(track)

    def send_track_states(self):
        track_state = { "trackState": {
                'driver': self.tracks['driver'].paused,
                'road': self.tracks['road'].paused,
                'wideRoad': self.tracks['wideRoad'].paused,
            }
        }
        self.data_channel.send(json.dumps(track_state))


    def attach_event_handlers(self):
        print("Creating Data Channel")
        self.data_channel = self.pc.createDataChannel("data")

        @self.data_channel.on("open")
        def on_open():
            print("[INFO] Data channel is open")
            # Get track state. ex. paused
            self.send_track_states()


        @self.data_channel.on("message")
        def on_message(message):
            print(f"[INFO] Received message: {message}")
            json_message = json.loads(message)
            print(f'{json_message=}')
            action = json_message.get("action", None)
            track_type = json_message.get("trackType", None)
            if action == "startTrack":
                if track_type in self.tracks.keys():
                    self.tracks[track_type].paused = False
            elif action == "stopTrack":
                if track_type in self.tracks.keys():
                    self.tracks[track_type].paused = True
            self.send_track_states()

        @self.data_channel.on("close")
        async def on_close():
            print("[INFO] Data channel closed")
            await self.stop()

        # async def on_icecandidate(event):
        #     if event.candidate:
        #         candidate = {
        #             'candidate': event.candidate.to_sdp(),
        #             'sdpMid': event.candidate.sdpMid,
        #             'sdpMLineIndex': event.candidate.sdpMLineIndex,
        #         }
        #         message = json.dumps({
        #             'type': 'candidate',
        #             'candidate': candidate,
        #         })
        #         self.ice_send_queue.put_nowait(message)
        #         print("[DEBUG] Sent ICE candidate")

        async def on_icegatheringstatechange():
            if self.pc:
                state = self.pc.iceGatheringState
                print(f"ice gathering state change: {state}")

        async def on_connectionstatechange():
            if self.pc:
                state = self.pc.connectionState
                print(f"connection state change: {state}")

        async def on_negotiationneeded():
            print("Negotiation needed - creating a new offer")
            await create_offer(self.pc)

        async def on_iceconnectionstatechange():
            if self.pc:
                print(f"ice connection state change: {self.pc.iceConnectionState}")

        #self.pc.on("icecandidate", on_icecandidate)
        self.pc.on("icegatheringstatechange", on_icegatheringstatechange)
        self.pc.on("connectionstatechange", on_connectionstatechange)
        self.pc.on("negotiationneeded", on_negotiationneeded)
        self.pc.on("iceconnectionstatechange", on_iceconnectionstatechange)

    async def setup_pc(self):

        iceServers = [
            RTCIceServer(
                urls="turn:85.190.241.173:3478",
                username="testuser",
                credential="testpass"
            ),
        ]
        configuration: RTCConfiguration = RTCConfiguration(iceServers=iceServers)
        print(configuration)
        self.pc = RTCPeerConnection(configuration)
        self.add_tracks()

    async def build(self):
        await self.setup_pc()
        self.attach_event_handlers()
        await create_offer(self.pc)
        while not self.pc.localDescription:
            await asyncio.sleep(0.1)
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)
        print(self.pc.localDescription.sdp)
        filtered_sdp = "\r\n".join(
            line for line in self.pc.localDescription.sdp.splitlines()
            if not line.startswith("a=candidate") or "relay" in line
        ) + "\r\n"
        message = json.dumps({
            'type': self.pc.localDescription.type,
            'sdp': self.pc.localDescription.sdp,
        })
        self.sdp_send_queue.put_nowait(message)
        print(f"Added Offer to send_queue {message}")

    async def stop(self):
        while not self.sdp_send_queue.empty(): # drain the queue
            self.sdp_send_queue.get()
        print("Closing PeerConnections")
        if self.pc:
            await self.pc.close()
        self.pc = None
        print("Stopping Camera")
        self.camera.stop()
        print("Stopping Encoder")
        self.encoder.stop()
        await asyncio.sleep(1) # wait for procs to stop

    async def event_loop(self, end_event: threading.Event):
        self.camera = NativeProcess("camerad", "system/camerad", ["./camerad"], True)
        self.encoder = NativeProcess("encoderd", "system/loggerd", ["./encoderd", "--stream"], True)
        stop_states = ['failed', 'closed']
        while not end_event.is_set():
            try:
                try:
                    data = self.sdp_recv_queue.get_nowait()
                except queue.Empty:
                    data = None

                if data:
                    print(f"[DEBUG] Got data in sdp_recv_queue: {data}")
                    msg_type = data.get('type')
                    if msg_type == 'start':
                        print("starting")
                        try:
                            await asyncio.wait_for(self.build(), timeout=30)
                        except asyncio.TimeoutError:
                            await self.stop()
                        self.camera.start() # Repeated calls are ok
                        self.encoder.start()
                        print("started")
                    elif msg_type == 'answer':
                        await set_answer(self.pc, data)
                    elif msg_type == 'candidate':
                        if 'candidate' in data:
                            await set_candidate(self.pc, data['candidate'])
                    elif msg_type == 'bye':
                        await self.stop()
                else:
                    await asyncio.sleep(0.1)

                # If connection failed, recreate the PC as needed.
                if self.pc:
                    transeivers = self.pc.getTransceivers()
                    dtls_state = None
                    if len(transeivers):
                        dtls_state = transeivers[0].receiver.transport.state

                    if self.pc.connectionState in stop_states or dtls_state in stop_states:
                        print("Connection failed - recreating PeerConnection")
                        await self.stop()

            except Exception as e:
                print(e)
                await self.stop()
        await self.stop()
