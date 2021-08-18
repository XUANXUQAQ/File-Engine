package file.engine.configs;

import java.net.Proxy;

public class ProxyInfo {
    public final String address;
    public final int port;
    public final String userName;
    public final String password;
    public final Proxy.Type type;

    public ProxyInfo(String proxyAddress, int proxyPort, String proxyUserName, String proxyPassword, int proxyType) {
        this.address = proxyAddress;
        this.port = proxyPort;
        this.userName = proxyUserName;
        this.password = proxyPassword;
        if (Constants.Enums.ProxyType.PROXY_HTTP == proxyType) {
            this.type = Proxy.Type.HTTP;
        } else if (Constants.Enums.ProxyType.PROXY_SOCKS == proxyType) {
            this.type = Proxy.Type.SOCKS;
        } else {
            this.type = Proxy.Type.DIRECT;
        }
    }
}
