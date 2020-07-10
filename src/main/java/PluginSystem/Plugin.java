package PluginSystem;

import javax.swing.*;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.util.concurrent.ConcurrentLinkedQueue;

public abstract class Plugin {
    public static class MessageStruct {
        public String message;
        public String caption;

        MessageStruct(String _message, String _caption) {
            this.message = _message;
            this.caption = _caption;
        }
    }

    private ConcurrentLinkedQueue<String> resultQueue = new ConcurrentLinkedQueue<>();
    private ConcurrentLinkedQueue<MessageStruct> messageQueue = new ConcurrentLinkedQueue<>();

    public abstract void textChanged(String text);

    protected void addToResultQueue(String result) {
        resultQueue.add(result);
    }

    public abstract void loadPlugin();

    public abstract void unloadPlugin();

    public String pollFromResultQueue() {
        return resultQueue.poll();
    }

    public abstract void keyReleased(KeyEvent e);

    public abstract void keyPressed(KeyEvent e);

    public abstract void keyTyped(KeyEvent e);

    public abstract void mousePressed(MouseEvent e);

    public abstract void mouseReleased(MouseEvent e);

    public abstract ImageIcon getPluginIcon();

    public abstract String getOfficialSite();

    public abstract String getVersion();

    public abstract String getDescription();

    protected void displayMessage(String message, String caption) {
        MessageStruct messages = new MessageStruct(message, caption);
        messageQueue.add(messages);
    }

    public MessageStruct getMessage() {
        return messageQueue.poll();
    }

    public abstract boolean isLatest();

    public abstract String getUpdateURL();
}
