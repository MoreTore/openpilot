import asyncio
import threading
import json
import logging
import queue
import requests

from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription, RTCIceCandidate, RTCRtpCodecCapability, RTCIceServer
from openpilot.system.webrtc.device.video import LiveStreamVideoStreamTrack

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

async def create_offer(pc, send_queue: queue.Queue):
    try:
        print("creating offer")
        offer = await pc.createOffer()
        print(f"Created Offer {offer}")
        await pc.setLocalDescription(offer)
        print("Set Local Description")
    except Exception as e:
        print(f"Failed to create offer: {e}")
        await pc.close()

async def set_answer(pc, data, send_queue: queue.Queue):
    try:
        answer = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
        await pc.setRemoteDescription(answer)
        print("Set remote description (answer)")
    except Exception as e:
        print(f"Failed to set answer: {e}")
        await pc.close()

async def set_candidate(pc, candidate_data, send_queue: queue.Queue):
    candidate = RTCIceCandidate(
        component=1,
        foundation=candidate_data.get("foundation", "1"),
        ip=candidate_data.get("ip", "0.0.0.0"),
        port=candidate_data.get("port", 0),
        priority=candidate_data.get("priority", 0),
        protocol=candidate_data.get("protocol", "udp"),
        type=candidate_data.get("type", "host"),
        relatedAddress=candidate_data.get("relatedAddress"),
        relatedPort=candidate_data.get("relatedPort"),
        sdpMid=candidate_data.get("sdpMid"),
        sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
        tcpType=candidate_data.get("tcpType")
    )
    await pc.addIceCandidate(candidate)

def attach_event_handlers(pc: RTCPeerConnection, send_queue: queue.Queue):
    async def on_icecandidate(event):
        if event.candidate:
            candidate = {
                'candidate': event.candidate.to_sdp(),
                'sdpMid': event.candidate.sdpMid,
                'sdpMLineIndex': event.candidate.sdpMLineIndex,
            }
            message = json.dumps({
                'type': 'candidate',
                'candidate': candidate,
            })
            send_queue.put_nowait(message)
            print("[DEBUG] Sent ICE candidate")

    async def on_icegatheringstatechange():
        state = pc.iceGatheringState
        print(f"ice gathering state change: {state}")
        if state == "complete" and pc.localDescription:
            message = json.dumps({
                'type': pc.localDescription.type,
                'sdp': pc.localDescription.sdp,
            })
            send_queue.put_nowait(message)
            print(f"Added Offer to send_queue {message}")

    async def on_connectionstatechange():
        state = pc.connectionState
        print(f"connection state change: {state}")
        # If connection fails, you might need to tear down and recreate the PC manually.

    async def on_negotiationneeded():
        print("Negotiation needed - creating a new offer")
        await create_offer(pc, send_queue)

    pc.on("icecandidate", on_icecandidate)
    pc.on("icegatheringstatechange", on_icegatheringstatechange)
    pc.on("connectionstatechange", on_connectionstatechange)
    pc.on("negotiationneeded", on_negotiationneeded)

async def setup_pc(send_queue: queue.Queue):
    configuration = RTCConfiguration(iceServers=get_ice_servers())
    pc = RTCPeerConnection(configuration)
    add_track("camera_type", pc)
    attach_event_handlers(pc, send_queue)
    return pc

async def webrtc_event_loop(end_event: threading.Event, send_queue: queue.Queue, sdp_recv_queue: queue.Queue):
    pc = await setup_pc(send_queue)
    await create_offer(pc, send_queue)

    while not end_event.is_set():
        try:
            data = sdp_recv_queue.get_nowait()
        except queue.Empty:
            data = None

        if data:
            msg_type = data.get('type')
            if msg_type == 'answer':
                await set_answer(pc, data, send_queue)
            elif msg_type == 'candidate':
                if 'candidate' in data:
                    await set_candidate(pc, data['candidate'], send_queue)
            elif msg_type == 'bye':
                await pc.close()
                return
        else:
            await asyncio.sleep(0.1)

        # If connection failed, recreate the PC as needed.
        if pc.connectionState == "failed":
            print("Connection failed - recreating PeerConnection")
            await pc.close()
            pc = await setup_pc(send_queue)
            await create_offer(pc, send_queue)

