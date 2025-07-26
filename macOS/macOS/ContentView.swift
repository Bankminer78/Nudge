import SwiftUI

struct ContentView: View {
    // The state now holds an array of UpdateCategory
    @State private var updates: [UpdateCategory] = []
    let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    
    var body: some View {
        VStack {
            Text("Your Updates")
                .font(.largeTitle)
                .padding()
            
            // Show a message if no updates have been loaded yet
            if updates.isEmpty {
                Spacer()
                Text("Checking for new updates...")
                    .font(.title2)
                    .foregroundColor(.secondary)
                Spacer()
            } else {
                // List the categories of updates
                List(updates) { category in
                    // Use a DisclosureGroup to make each category expandable
                    DisclosureGroup {
                        // List the clickable links for each item in the category
                        ForEach(category.items) { item in
                            Link(item.title, destination: item.link)
                                .padding(.leading, 15) // Indent items for clarity
                                .onHover { hovering in
                                    if hovering {
                                        NSCursor.pointingHand.push()
                                    } else {
                                        NSCursor.pop()
                                    }
                                }
                        }
                    } label: {
                        Text(category.title)
                            .font(.headline)
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .onReceive(timer) { _ in
            checkForUpdates()
        }
    }
    
    func checkForUpdates() {
        MockData.checkForUpdates { newUpdates in
            // When new updates are found, update the state and send a notification
            if let newUpdates = newUpdates {
                self.updates = newUpdates
                
                // Make the notification body more descriptive
                let titles = newUpdates.map { $0.title }.joined(separator: ", ")
                NotificationManager.shared.sendNotification(
                    title: "New Updates Available!",
                    body: "We found new items for: \(titles)."
                )
            }
        }
    }
}

// The preview provider remains the same
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
