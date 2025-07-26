import Foundation
import AppKit
import UserNotifications

class NotificationManager: NSObject, UNUserNotificationCenterDelegate {
    static let shared = NotificationManager()
    
    /// Requests user permission to send notifications.
    func requestAuthorization() {
        let center = UNUserNotificationCenter.current()
        center.requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if let error = error {
                print("Error requesting notification authorization: \(error)")
            }
        }
    }
    
    /// Creates and sends a local notification.
    func sendNotification(title: String, body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        
        let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }

    /// **NEW**: Sends a welcome notification immediately.
    /// Call this after the app has launched to show a startup notification.
    func sendWelcomeNotification() {
        let content = UNMutableNotificationContent()
        content.title = "Update Checker Ready!"
        content.body = "Monitoring your favorite topics for new updates."
        content.sound = .default
        
        // Trigger immediately
        let request = UNNotificationRequest(identifier: "welcomeNotification", content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("Error sending welcome notification: \(error.localizedDescription)")
            }
        }
    }
    
    /// Delegate method for handling notifications that arrive while the app is open.
    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification, withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        // Show the alert, play a sound, and update the badge icon.
        completionHandler([.alert, .sound, .badge])
    }
    
    /// Delegate method for handling when a user clicks on a notification.
    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse, withCompletionHandler completionHandler: @escaping () -> Void) {
        // Handle the user clicking on the notification.
        print("Notification clicked! Bringing app to the foreground.")
        
        // Bring the app to the front.
        NSApp.activate(ignoringOtherApps: true)
        
        // Explicitly find the main window and make it the key window.
        // This ensures the content view becomes visible, even if the app was just hidden.
        if let window = NSApp.windows.first {
            window.makeKeyAndOrderFront(nil)
        }
        
        completionHandler()
    }
}
