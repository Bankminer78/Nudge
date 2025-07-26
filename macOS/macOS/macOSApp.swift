//
//  macOSApp.swift
//  macOS
//
//  Created by Tanish Pradhan Wong Ah Sui on 7/26/25.
//

import SwiftUI
import UserNotifications
import UserNotifications

@main
struct macOSApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ aNotification: Notification) {
        NotificationManager.shared.requestAuthorization()
        UNUserNotificationCenter.current().delegate = NotificationManager.shared
    }
}
