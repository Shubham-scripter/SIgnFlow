import asyncio
import websockets


async def test_websocket():

    uri = "ws://127.0.0.1:8000/ws"

    print("Connecting to SignFlow backend...")

    async with websockets.connect(uri) as websocket:

        # Receive connection message
        response = await websocket.recv()

        print("\nBackend says:")
        print(response)

        # Send ping
        await websocket.send(
            '{"type": "ping"}'
        )

        # Receive response
        response = await websocket.recv()

        print("\nBackend response:")
        print(response)


asyncio.run(test_websocket())