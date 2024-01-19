package file.engine.services.utils;

import lombok.extern.slf4j.Slf4j;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;

@Slf4j
public class CoreUtil {

    public static String getCoreResult(String method, String path, int port) {
        try (DatagramSocket socket = new DatagramSocket()) {
            socket.setSoTimeout(1000);
            String url = method + " " + path;
            byte[] bytes = url.getBytes(StandardCharsets.UTF_8);
            DatagramPacket datagramPacket = new DatagramPacket(bytes, bytes.length, InetAddress.getLocalHost(), port);
            socket.send(datagramPacket);
            byte[] res = new byte[4];
            DatagramPacket lenPacket = new DatagramPacket(res, res.length);
            socket.receive(lenPacket);
            int len = byte4ToInt(lenPacket.getData());
            byte[] result = new byte[len];
            DatagramPacket resultPacket = new DatagramPacket(result, len);
            socket.receive(resultPacket);
            return new String(resultPacket.getData(), resultPacket.getOffset(), resultPacket.getLength(), StandardCharsets.UTF_8);
        } catch (Exception e) {
            log.error(e.getMessage(), e);
            return "";
        }
    }

    public static String get(String path, int port) {
        return getCoreResult("GET", path, port);
    }

    public static String post(String path, int port) {
        return getCoreResult("POST", path, port);
    }

    private static int byte4ToInt(byte[] bytes) {
        int b0 = bytes[0] & 0xFF;
        int b1 = bytes[1] & 0xFF;
        int b2 = bytes[2] & 0xFF;
        int b3 = bytes[3] & 0xFF;
        return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    }
}
