import SwiftUI

struct ContentView: View {
    @State private var updates: [Update] = []
    let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    
    var body: some View {
        VStack {
            Text("Updates")
                .font(.largeTitle)
                .padding()
            
            List(updates) { update in
                VStack(alignment: .leading) {
                    Text(update.title)
                        .font(.headline)
                    Text(update.description)
                        .font(.subheadline)
                }
            }
        }
        .onReceive(timer) { _ in
            checkForUpdates()
        }
    }
    
    func checkForUpdates() {
        MockData.checkForUpdates { newUpdates in
            if let newUpdates = newUpdates {
                self.updates = newUpdates
                NotificationManager.shared.sendNotification(title: "New Updates Available", body: "Check out the latest changes.")
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
